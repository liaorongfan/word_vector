import os
import pickle
import torch
from lib.data_process import clean_sentence, cut_words
from lib.config import Config


class WordDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.config = config
        self.file_path = config.file_path
        self.stop_words_file = config.stop_words_file
        self.context_size = config.context_size
        self.device = config.device

        self.raw_data = self.collect_words()
        self.__samples = self.make_cbow_data()

        self.vocab = set(self.raw_data)
        self.vocab_size = len(self.vocab)

        self.word_to_idx = {word: i for i, word in enumerate(self.vocab)}
        self.idx_to_word = {i: word for i, word in enumerate(self.vocab)}
        self.save_word_index()

    def __len__(self):
        return len(self.__samples)

    def __getitem__(self, idx):
        context, target = self.__samples[idx]
        context_vector, target_idx = self._transform(context, target)
        return context_vector, target_idx

    def collect_words(self):
        with open(self.file_path, encoding='utf8') as fo:
            allData = fo.readlines()
        with open(self.stop_words_file, "r", encoding="utf-8") as fo:
            stop_words = fo.read().split("\n")

        corpus = []
        for words in allData:
            words = clean_sentence(words)
            words = cut_words(words, stop_words)
            corpus.extend(words)

        return corpus

    def make_cbow_data(self):
        samples = []
        cs = self.context_size
        for i in range(cs, len(self.raw_data) - cs):
            context = self.raw_data[i - cs:i] + self.raw_data[i + 1:i + cs + 1]
            target = self.raw_data[i]
            samples.append((context, target))
        return samples

    def _transform(self, context, target):
        con_idx = [self.word_to_idx[w] for w in context]
        tar_idx = self.word_to_idx[target]
        con_idx = torch.tensor(con_idx, dtype=torch.long).to(self.device)
        tar_idx = torch.tensor([tar_idx], dtype=torch.long).to(self.device)
        return con_idx, tar_idx

    def get_word2idx(self):
        return self.word_to_idx

    def get_idx2word(self):
        return self.idx_to_word

    def save_word_index(self):
        save_to = os.path.join(
            self.config.output_dir, self.config.word_to_idx_file)
        with open(save_to, 'wb') as fo:
            pickle.dump(self.word_to_idx, fo)
        print(f">>> word_to_idx saved in {save_to}")

        save_to = os.path.join(
            self.config.output_dir, self.config.idx_to_word_file)
        with open(save_to, 'wb') as fo:
            pickle.dump(self.idx_to_word, fo)
        print(f">>> idx_to_word saved in {save_to}")
