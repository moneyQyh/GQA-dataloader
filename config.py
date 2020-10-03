

qa_path = '/home_export/qyx/VQA-X/Annotations'  # directory containing the question and annotation jsons
gqa_path = '/home_export/qyx/GQA/balanced_data/'
bottom_up_trainval_path = '/home/qyx/datasets/vqa/trainval_36'  # directory containing the .tsv file(s) with bottom up features
bottom_up_test_path = '/home_export/qyx/datasets/test2015_36'  # directory containing the .tsv file(s) with bottom up features
preprocessed_trainval_path = '/home_export/qyx/VQA-X/Rcnn_output'  # path where preprocessed features from the trainval split are saved to and loaded from
gqa_preprocessed_trainval_path = '/home_export/qyx/GQA/images/gqa_objects.hdf5'  # path where preprocessed features from the trainval split are saved to and loaded from
gqa_info_path = '/home_export/qyx/GQA/images/gqa_objects_merged_info.json'
preprocessed_test_path = '/home_export/qyx/VQA-X/Rcnn_test2015_output'  # path where preprocessed features from the test split are saved to and loaded from
question_vocabulary_path = 'question_vocab.json'  # path where the used vocabularies for question and answers are saved to
answer_vocabulary_path = 'answer_vocab.json'  # path where the used vocabularies for question and answers are saved to
explanation_vocabulary_path = 'exp_vocab.json'  # path where the used vocabularies for question and answers are saved to
gqa_question_vocabulary_path = 'gqa_question_vocab.json'  # path where the used vocabularies for question and answers are saved to
gqa_answer_vocabulary_path = 'gqa_answer_vocab.json'  # path where the used vocabularies for question and answers are saved to
gqa_explanation_vocabulary_path = 'gqa_exp_vocab.json'  # path where the used vocabularies for question and answers are saved to
caption_vocabulary_path ='caption_vocab.json'
filter_caption_path = 'filter_caption.json'
gqadataset ='GQA/balanced_data'
GQA_path = '/home_export/qyx/GQA/balanced_data'
test_split = 'test'  # either 'test-dev2015' or 'test2015'