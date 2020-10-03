import argparse
import json
from collections import Counter
import itertools
import utils_gqa
import config
import data
import os
import data_gqa
def extract_vocab(iterable, top_k=None, start=0):
    """ Turns an iterable of list of tokens into a vocabulary.
        These tokens could be single answers or word tokens in questions.
    """
    all_tokens = itertools.chain.from_iterable(iterable)
    counter = Counter(all_tokens)
    if top_k:
        most_common = counter.most_common(top_k)
        most_common = (t for t, c in most_common)
    else:
        most_common = counter.keys()
    # descending in count, then lexicographical order
    tokens = sorted(most_common, key=lambda x: (counter[x], x), reverse=True)
    vocab = {t: i+1 for i, t in enumerate(tokens)}
    vocab['<unk>'] = len(vocab) + 2
    vocab['<start>'] = len(vocab) + 3
    vocab['<end>'] = len(vocab) + 4
    vocab['<pad>'] = 0
    return vocab


def main():
    # questions = utils.path_for(train=True, question=True)
    # answers = utils.path_for(train=True, answer=True)
    # explanations = utils.path_for(train=True,explanation=True)

    # captions = utils.path_for(train=True,explanation=True)
    #
    data_path = utils_gqa.path_for(train=True, test=False, val=False)
    with open(data_path,'r') as fd:
        data = json.load(fd)


    # with open(captions,'r') as fd:
    #     captions = json.load(fd)
    questions = [data_item['question'] for qid, data_item in data.items()]
    answers = [data_item['answer'] for qid, data_item in data.items()]
    explanations = [data_item['fullAnswer'] for qid, data_item in data.items()]
    questions = list(data_gqa.prepare_questions(questions))
    answers = list(data_gqa.prepare_answers(answers))
    explanations =list(data_gqa.prepare_explanation(explanations))
    #captions = data.prepare_captions(caption)
    question_vocab = extract_vocab(questions, start=0)
    answer_vocab = extract_vocab(answers, top_k=config.max_answers)
    explanations_vocab = extract_vocab(explanations,start=0)
    #captions_vocab = extract_vocab(captions)

   # vocabs = {
    #    'question': question_vocab,
    #}
    with open(config.gqa_question_vocabulary_path, 'w') as fd:
        json.dump(question_vocab, fd)
    with open(config.gqa_answer_vocabulary_path, 'w') as fd:
        json.dump(answer_vocab, fd)
    with open(config.gqa_explanation_vocabulary_path, 'w') as fd:
        json.dump(explanations_vocab, fd)



if __name__ == '__main__':
    main()
