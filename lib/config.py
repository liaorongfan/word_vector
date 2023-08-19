import os
import torch


class Config:
    # env
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data
    file_path = 'data/corpus/agriculture_policy.txt'
    stop_words_file = 'data/corpus/stopwords.txt'
    context_size = 3

    # model
    embedding_dim = 256
    model_file = "cbow.pth"

    # train
    epoch = 30
    lr = 0.01

    # output
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    word_to_vec_file = "agriculture_wordvec_cbow.pkl"
    word_to_vec_dict_file = "agriculture_wordvec_dict_cbow.pkl"
    word_to_idx_file = "word2idx.pkl"
    idx_to_word_file = "idx2word.pkl"

