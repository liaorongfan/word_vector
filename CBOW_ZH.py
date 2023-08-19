"""
https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html?highlight=cbow
"""
import pickle

import jieba
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm, trange
import re
from pylab import mpl

torch.manual_seed(1)


def remove_number(sentence):
    cleaned_sentence = re.sub(r'\d+', '', sentence)
    return cleaned_sentence


def remove_character(sentence):
    cleaned_sentence = re.sub(r'[a-zA-Z]', '', sentence)
    return cleaned_sentence


def remove_special_character(sentence):
    cleaned_sentence = re.sub(r'[^\w\s]', '', sentence)
    return cleaned_sentence


def clean_sentence(sentence):
    cleaned_sentence = remove_number(sentence)
    cleaned_sentence = remove_character(cleaned_sentence)
    cleaned_sentence = remove_special_character(cleaned_sentence)
    cleaned_sentence = cleaned_sentence.replace(" ", "")
    cleaned_sentence = cleaned_sentence.replace("\n", "")
    return cleaned_sentence


def cut_words(sentence, stop_words):
    c_words = jieba.lcut(sentence)
    return [w for w in c_words if w not in stop_words]


# 加载文本,切词
def collect_words(
    file_path='data/corpus/agriculture_policy.txt',
    stop_words_file='data/corpus/stopwords.txt',
):
    with open(file_path, encoding='utf8') as fo:
        allData = fo.readlines()
    with open(stop_words_file, "r", encoding="utf-8") as fo:
        stop_words = fo.read().split("\n")

    result = []
    for words in allData:
        words = clean_sentence(words)
        words = cut_words(words, stop_words)
        result.extend(words)

    return result


def make_cbow_data(raw_data, context_size=2):
    data = []
    cs = context_size
    for i in range(cs, len(raw_data) - cs):
        context = raw_data[i - cs:i] + raw_data[i + 1:i + cs + 1]
        target = raw_data[i]
        data.append((context, target))
    return data


class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.proj = nn.Linear(embedding_dim, 2048)
        self.output = nn.Linear(2048, vocab_size)

    def forward(self, inputs):
        embeds = sum(self.embeddings(inputs)).view(1, -1)
        out = F.relu(self.proj(embeds))
        out = self.output(out)
        nll_prob = F.log_softmax(out, dim=-1)
        return nll_prob


def make_context_vector(context,  word_to_ix, device):
    con_idx = [word_to_ix[w] for w in context]
    con_idx = torch.tensor(con_idx, dtype=torch.long).to(device)
    return con_idx


def convert_training_sample(context, target, word_to_ix, device):
    con_idx = [word_to_ix[w] for w in context]
    tar_idx = word_to_ix[target]
    con_idx = torch.tensor(con_idx, dtype=torch.long).to(device)
    tar_idx = torch.tensor([tar_idx], dtype=torch.long).to(device)
    return con_idx, tar_idx


def train(model, dataloader, epoch=10, lr=0.001, device="cuda"):
    optimizer = optim.SGD(model.parameters(), lr)
    loss_function = nn.NLLLoss()

    for epo in trange(epoch):
        total_loss, i = 0, 0
        for context, target in tqdm(dataloader):
            model.zero_grad()
            train_predict = model(context).to(device)
            loss = loss_function(train_predict, target)
            loss.backward()
            optimizer.step()
            if i % 1000 == 0:
                print("batch loss = ", loss.item())
            total_loss += loss.item()
            i += 1
        print("epo losses = ", total_loss / len(dataloader))
    torch.save(model.state_dict(), 'cbow.pth')
    return model


def show_word_proj_plot(W, word_to_idx):

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(W)

    # 降维后在生成一个词嵌入字典，即即{单词1:(维度一，维度二),单词2:(维度一，维度二)...}的格式
    word2ReduceDimensionVec = {}
    for word in word_to_idx.keys():
        word2ReduceDimensionVec[word] = principalComponents[word_to_idx[word], :]

    # 将词向量可视化
    plt.figure(figsize=(40, 40))
    # 只画出1000个，太多显示效果很差
    count = 0
    for word, wordvec in word2ReduceDimensionVec.items():
        if count < 500:
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
            plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号，否则负号会显示成方块
            plt.scatter(wordvec[0], wordvec[1], s=50)
            plt.annotate(word, (wordvec[0], wordvec[1]), fontsize=16)
            count += 1
    plt.savefig("CBOW_ZH_wordvec.png", dpi=400)


def predict_word(model, context, word_to_idx, idx_to_word, device):
    context_vector = make_context_vector(context, word_to_idx, device)
    predict = model(context_vector).data.cpu().numpy()
    max_idx = np.argmax(predict)
    print('Prediction: {}'.format(idx_to_word[max_idx]))


def save_wordvec_dict(wordvec, word_to_idx):
    # 生成词嵌入字典，即{单词1:词向量1,单词2:词向量2...}的格式
    word_2_vec = {}
    for word in word_to_idx.keys():
        # 词向量矩阵中某个词的索引所对应的那一列即为所该词的词向量
        word_2_vec[word] = wordvec[word_to_idx[word], :]
    with open("agriculture_wordvec_dict_cbow.pkl", 'wb') as f:
        pickle.dump(word_2_vec, f)

    print(">>> word_vector_dict saved in agriculture_wordvec_dict_cbow.pkl")


class WordDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        file_path='data/corpus/agriculture_policy.txt',
        stop_words_file='data/corpus/stopwords.txt',
        context_size=2,
        device=torch.device("cuda"),
    ):
        self.file_path = file_path
        self.stop_words_file = stop_words_file
        self.context_size = context_size
        self.device = device

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
        with open("word2idx.pkl", 'wb') as fo:
            pickle.dump(self.word_to_idx, fo)
        with open("idx2word.pkl", 'wb') as fo:
            pickle.dump(self.idx_to_word, fo)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # raw_data = collect_words()
    # vocab = set(raw_data)
    # vocab_size = len(vocab)
    #
    # word_to_idx = {word: i for i, word in enumerate(vocab)}
    # idx_to_word = {i: word for i, word in enumerate(vocab)}

    # with open("word2idx.pkl", 'wb') as f:
    #     pickle.dump(word_to_idx, f)
    # with open("idx2word.pkl", 'wb') as f:
    #     pickle.dump(idx_to_word, f)
    #
    # data = make_cbow_data(raw_data, context_size=2)
    word_dataset = WordDataset(device=device)
    vocab_size = word_dataset.vocab_size
    word_to_idx = word_dataset.get_word2idx()

    cbow = CBOW(vocab_size, embedding_dim=256).to(device)
    model = train(cbow, word_dataset, epoch=300, lr=0.01)

    W = model.embeddings.weight.cpu().detach().numpy()
    with open("agriculture_wordvec_cbow.pkl", 'wb') as f:
        pickle.dump(W, f)
    print(">>> word_vector saved in agriculture_wordvec_cbow.pkl")

    save_wordvec_dict(W, word_to_idx)

