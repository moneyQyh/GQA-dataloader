import json
import os
import os.path
import re

from PIL import Image
import h5py
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import config
import utils
import utils_gqa
import collections

question_ids = list()
coco_ids =list()

def get_loader(train=False, val=False, test=False,on_pretrain=False):
    """ Returns a data loader for the desired split """
    split = GQA(
        utils_gqa.path_for(train=train, val=val, test=test),
        config.gqa_info_path,
        config.gqa_preprocessed_trainval_path if not test else config.preprocessed_test_path,
        on_pretrain= on_pretrain,
        answerable_only=train,
        dummy_answers=test,
    )
    loader = torch.utils.data.DataLoader(
        split,
        batch_size=config.batch_size,
        shuffle=train,  # only shuffle the data in training
        pin_memory=True,
        num_workers=config.data_workers,
        drop_last = True,
        #collate_fn=collate_fn,
    )
    return loader
class GQA(data.Dataset):
    """ VQA dataset, open-ended """
    def __init__(self, content_path, image_info_path ,image_features_path,on_pretrain=False,answerable_only=False, dummy_answers=False,exp_train_mode = True):
        super(GQA, self).__init__()

        with open(content_path, 'r') as fd:
            content_json = json.load(fd)


        # with open('dataset_coco.json', 'r') as j:
        #     captions_json = json.load(j)
       # Load caption lengths (completely into memory)
       #  with open(caption_path, 'r') as j:
       #      self.caplens = json.load(j)
        # current location put token2word
        with open('gqa_ques_token2word.json','r')  as fd:
            check_ques_json = json.load(fd)
        with open('gqa_exp_token2word.json','r')  as fd:
            check_exp_json = json.load(fd)
        self.on_pretrain = on_pretrain
        # if preloaded_vocab:
        #     vocab_json = preloaded_vocab

        with open(config.gqa_question_vocabulary_path, 'r') as fd:
                question_vocab_json = json.load(fd)
        with open(config.gqa_answer_vocabulary_path, 'r') as fd:
                answer_vocab_json = json.load(fd)
        with open(config.gqa_explanation_vocabulary_path, 'r') as fd:
                explanation_vocab_json = json.load(fd)

            # with open(config.caption_vocabulary_path, 'r') as fd:
            #      caption_vocab_json = json.load(fd)
        self.question_ids = [qid for qid,content in content_json.items()]

        question_ids = self.question_ids
        #索引序列编写
        self.token2ques = check_ques_json
        self.token2exp = check_exp_json


        questions_json = [data_item['question'] for qid, data_item in content_json.items()] #获取全部的问题
        answers_json = [data_item['answer'] for qid, data_item in content_json.items()] #获取全部的答案
        explanation_json = [data_item['fullAnswer'] for qid, data_item in content_json.items()] #获取全部的解释

        self.exp_train_mode = exp_train_mode
        # vocab
        self.question_vocab = question_vocab_json
        self.answer_vocab = answer_vocab_json

        self.vocab = {'question':self.question_vocab,'answer':self.answer_vocab}
        self.explanation_vocab = explanation_vocab_json
        # self.caption_vocab = caption_vocab_json
        self.token_to_index = self.question_vocab
        self.answer_to_index = self.answer_vocab
        self.exp_to_index = self.explanation_vocab
        # self.cap_to_index = self.caption_vocab
        # self.check_index = check_exp_json
        # v
        self.image_info_path = image_info_path
        self.image_features_path = image_features_path
        print('reading image')
        self.coco_id_to_index = self._create_coco_id_to_index()
         # q and a
        if on_pretrain:
           self.questions = list(prepare_questions(questions_json))
           self.answers = list(prepare_answers(answers_json))
           self.questions = [self._encode_question(q) for q in self.questions]
           self.answers = [self._encode_answers(a) for a in self.answers]

        self.coco_ids = [data_item['imageId'] for qid, data_item in content_json.items()]
         # self.coco_ids_set = list(set(self.coco_ids))
         # self.coco_ids_set.sort(key=self.coco_ids.index)
        self.dummy_answers= dummy_answers

        # cpation
        # print('reading captions')
        # self.first_training = False
        # #self.caption_ids = self.get_captionid(captions_json)
        # if self.first_training:
        #    self.filter_captions= self.filter_caption(self.coco_ids_set, captions_json)
        #    with open(config.filter_caption_path, 'w') as fd:
        #        json.dump(self.filter_captions, fd)
        # else:
        #     with open(config.filter_caption_path, 'r') as j:
        #         self.filter_captions = json.load(j)
        #
        # print('process all captions')
        # self.captions = [self._encode_captions(c) for c in self.filter_captions]
        #exp

        if not on_pretrain:
            self.questions = list(prepare_questions(questions_json))
            self.answers = list(prepare_answers(answers_json))
            self.explanations = list(prepare_explanation(explanation_json))
            self.questions = [self._encode_question(q) for q in self.questions]
            self.answers = [self._encode_answers(a) for a in self.answers]
            self.explanations = [self._encode_explanation(e) for e in self.explanations]
            #print(self.explanations)



        # self.count = 0
        #filter

        # only use questions that have at least one answer?
        self.answerable_only = answerable_only
        if self.answerable_only:
            self.answerable = self._find_answerable(not self.answerable_only)


    @property
    def max_question_length(self):
        if not hasattr(self, '_max_length'):
            self._max_length = max(map(len, self.questions))
        return self._max_length


    @property
    def max_explantion_length(self):
        if not hasattr(self, 'explanation_max_length'):
             self.explanation_max_length = max(map(len,self.explanations))
        return self.explanation_max_length

    # @property
    # def max_caption_length(self):
    #     if not hasattr(self, '_max_length'):
    #         self._max_length = max(map(len, self.filter_captions))
    #     return self._max_length
    @property
    def num_tokens(self):
        return len(self.token_to_index) + 1  # add 1 for <unknown> token at index 0

    def _create_coco_id_to_index(self):
        """ Create a mapping from a COCO image id into the corresponding index into the h5 file """
        with open(self.image_info_path,'r') as fd:
            image_infos = json.load(fd)
        coco_id_to_index = {id: image_info['index'] for id, image_info in image_infos.items()}
        return coco_id_to_index

    def _check_integrity(self, questions, answers , captions):
        """ Verify that we are using the correct data """
        qa_pairs = list(zip(questions['questions'], answers['annotations']))
        assert all(q['question_id'] == a['question_id'] for q, a in qa_pairs), 'Questions not aligned with answers'
        assert all(q['image_id'] == a['image_id'] for q, a in qa_pairs), 'Image id of question and answer don\'t match'
        assert questions['data_type'] == answers['data_type'], 'Mismatched data types'
        assert questions['data_subtype'] == answers['data_subtype'], 'Mismatched data subtypes'

    def _find_answerable(self, count=False):
        """ Create a list of indices into questions that will have at least one answer that is in the vocab """
        answerable = []
        if count:
            number_indices = torch.LongTensor([self.answer_to_index[str(i)] for i in range(0, 8)])
        for i, answers in enumerate(self.answers):
            # store the indices of anything that is answerable
            if count:
                answers = answers[number_indices]
            answer_has_index = len(answers.nonzero()) > 0
            if answer_has_index:
                answerable.append(i)
        return answerable

    def _encode_explanation(self, explanation):
        """ Turn a explanation into a vector of indices and a question length """


        vec = torch.zeros(self.max_explantion_length+2).long()
        start = self.exp_to_index.get('<start>')
        end = self.exp_to_index.get('<end>')
        unk = self.exp_to_index.get('<unk>')
        vec[0] = start
        for i, token in enumerate(explanation):
            index = self.exp_to_index.get(token, unk)
            vec[i+1] = index
            flag =i+1
        vec[flag+1] = end
        #print(vec)
        return vec
    def _encode_question(self, question):
        """ Turn a question into a vector of indices and a question length """

        vec = torch.zeros(self.max_question_length).long()
        for i, token in enumerate(question):
            index = self.token_to_index.get(token,13759)
            vec[i] = index


        return vec, len(question)

    def _encode_answers(self, answers):
        """ Turn an answer into a vector """
        # answer vec will be a vector of answer counts to determine which answers will contribute to the loss.
        # this should be multiplied with 0.1 * negative log-likelihoods that a model produces and then summed up
        # to get the loss that is weighted by how many humans gave that answer
        answer_vec = torch.zeros(len(self.answer_to_index))
        for answer in answers:
            index = self.answer_to_index.get(answer)
            if index is not None:
                answer_vec[index] += 1
        return answer_vec

    def _load_image(self, image_id):
        """ Load an image """
        if not hasattr(self, 'features_file'):
            # Loading the h5 file has to be done here and not in __init__ because when the DataLoader
            # forks for multiple works, every child would use the same file object and fail
            # Having multiple readers using different file objects is fine though, so we just init in here.
            self.features_file = h5py.File(self.image_features_path, 'r')
            for key in self.features_file.keys():
                print(self.features_file[key].name)
                #print(f[key].shape)
                #print(f[key].value)

        index = self.coco_id_to_index[image_id]

        img = self.features_file['features'][index]
        boxes = self.features_file['bboxes'][index]
        obj_mask = (img.sum(0) > 0).astype(int)
        return torch.from_numpy(img), torch.from_numpy(boxes), torch.from_numpy(obj_mask)


    def check_ques_right(self, encode_conse):
        list_enco = []
        for i, token in enumerate(encode_conse):
            #print(token)
            if token ==0:
                continue
            list_enco.append(self.token2ques[str(token)])

        print(list_enco)

    def check_exp_right(self, encode_conse):

        list_enco = []
        for i, exps in enumerate(encode_conse):
            # print(token)
            if exps ==0:
                continue
            list_enco.append(self.token2exp[str(exps)])
        print(list_enco)
    def __getitem__(self, item):
        if self.answerable_only:
            item = self.answerable[item]
        #主要是这个item如何去理解
        q, q_length = self.questions[item]

        exp = self.explanations[item]


        # q = q.numpy().tolist()
        # exp = exp.numpy().tolist()
        # self.check_ques_right(q)
        # self.check_exp_right(exp)

        # caps,cap_lengths = self.captions[item]
        q_mask = torch.from_numpy((np.arange(self.max_question_length) < q_length).astype(int))


        # vec_caption_mask = []
        # for cap_length in cap_lengths:
        #   caption_mask = torch.from_numpy((np.arange(self.max_caption_length)<cap_length).astype(int))
        #   vec_caption_mask.append(caption_mask)
        if not self.dummy_answers:
            a = self.answers[item]
        else:
            # just return a dummy answer, it's not going to be used anyway
            a = 0
        image_id = self.coco_ids[item]
        # print(caps)
        v, b, obj_mask = self._load_image(image_id)

        # since batches are re-ordered for PackedSequence's, the original question order is lost
        # we return `item` so that the order of (v, q, a) triples can be restored if desired
        # without shuffling in the dataloader, these will be in the order that they appear in the q and a json's.

        return v, q, a, exp , b, item, obj_mask, q_mask,q_length
    def __len__(self):
        if self.answerable_only:
            return len(self.answerable)
        else:
            return len(self.questions)

# this is used for normalizing questions
_special_chars = re.compile('[^a-z0-9 ]*')

# these try to emulate the original normalization scheme for answers
_period_strip = re.compile(r'(?!<=\d)(\.)(?!\d)')
_comma_strip = re.compile(r'(\d)(,)(\d)')
_punctuation_chars = re.escape(r';/[]"{}()=+\_-><@`,?!')
_punctuation = re.compile(r'([{}])'.format(re.escape(_punctuation_chars)))
_punctuation_with_a_space = re.compile(r'(?<= )([{0}])|([{0}])(?= )'.format(_punctuation_chars))
def prepare_captions(captions_json):
   # print(coco_ids)
    captions = []
    for img in captions_json['images']:

        for c in img['sentences']:
            # Update word frequency
            #if len(c['tokens']) <= max_len:
                print(c['tokens'])
                captions.append(c['tokens'])
    return  captions

def prepare_filter_question(filter_q_dict):
    questions = [q['q_str'] for k,q in filter_q_dict]
    for question in questions:
        question = question.lower()[:-1]
        question = _special_chars.sub('', question)
        yield question.split(' ')
def prepare_filter_answers(filter_a_dict):
     answers = [[a['answer'] for a in ans] for qid, ans in filter_a_dict.items()]

     def process_punctuation(s):
         # the original is somewhat broken, so things that look odd here might just be to mimic that behaviour
         # this version should be faster since we use re instead of repeated operations on str's
         if _punctuation.search(s) is None:
             return s
         s = _punctuation_with_a_space.sub('', s)
         if re.search(_comma_strip, s) is not None:
             s = s.replace(',', '')
         s = _punctuation.sub(' ', s)
         s = _period_strip.sub('', s)
         return s.strip()

     for answer_list in answers:
         yield list(map(process_punctuation, answer_list))
def prepare_questions(questions):
    """ Tokenize and normalize questions from a given question json in the usual VQA format. """
    #questions = [q['question'] for q in questions_json['questions']]
    for question in questions:
        question = question.lower()[:-1]
        question = _special_chars.sub('', question)
        yield question.split(' ')
def prepare_explanation(explanations):
    for explanation in explanations:
        explanation = explanation.lower()[:-1]
        # print(explanation)
        explanation = _special_chars.sub('', explanation)
        yield explanation.split(' ')

def prepare_answers(answers):
    """ Normalize answers from a given answer json in the usual VQA format. """
    #answers = [[a['answer'] for a in ans_dict['answers']] for ans_dict in answers_json['annotations']]
    # The only normalization that is applied to both machine generated answers as well as
    # ground truth answers is replacing most punctuation with space (see [0] and [1]).
    # Since potential machine generated answers are just taken from most common answers, applying the other
    # normalizations is not needed, assuming that the human answers are already normalized.
    # [0]: http://visualqa.org/evaluation.html
    # [1]: https://github.com/VT-vision-lab/VQA/blob/3849b1eae04a0ffd83f56ad6f70ebd0767e09e0f/PythonEvaluationTools/vqaEvaluation/vqaEval.py#L96

    def process_punctuation(s):
        # the original is somewhat broken, so things that look odd here might just be to mimic that behaviour
        # this version should be faster since we use re instead of repeated operations on str's
        if _punctuation.search(s) is None:
            return s
        s = _punctuation_with_a_space.sub('', s)
        if re.search(_comma_strip, s) is not None:
            s = s.replace(',', '')
        s = _punctuation.sub(' ', s)
        s = _period_strip.sub('', s)
        return s.strip()

    for answer_list in answers:
        yield list(map(process_punctuation, answer_list))



class CocoImages(data.Dataset):
    """ Dataset for MSCOCO images located in a folder on the filesystem """
    def __init__(self, path, transform=None):
        super(CocoImages, self).__init__()
        self.path = path
        self.id_to_filename = self._find_images()
        self.sorted_ids = sorted(self.id_to_filename.keys())  # used for deterministic iteration order
        print('found {} images in {}'.format(len(self), self.path))
        self.transform = transform

    def _find_images(self):
        id_to_filename = {}
        for filename in os.listdir(self.path):
            if not filename.endswith('.jpg'):
                continue
            id_and_extension = filename.split('_')[-1]
            id = int(id_and_extension.split('.')[0])
            id_to_filename[id] = filename
        return id_to_filename

    def __getitem__(self, item):
        id = self.sorted_ids[item]
        path = os.path.join(self.path, self.id_to_filename[id])
        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return id, img

    def __len__(self):
        return len(self.sorted_ids)

te_loader = get_loader(train=True,on_pretrain=False)
for v, q, a, exp,b, idx, v_mask, q_mask, _ in te_loader:
    print(q_mask.shape)
    print(exp.shape)
    print(v.shape)
    print(q.shape)
    print(a.shape)
    print(b.shape)
