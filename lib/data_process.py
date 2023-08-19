import re
import jieba
import torch
import pickle
import os


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


def collect_words(file_path, stop_words_file):
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


def make_context_vector(context,  word_to_ix, device):
    con_idx = [word_to_ix[w] for w in context]
    con_idx = torch.tensor(con_idx, dtype=torch.long).to(device)
    return con_idx


def save_wordvec_dict(config, wordvec, word_to_idx):
    # 生成词嵌入字典，即{单词1:词向量1,单词2:词向量2...}的格式
    word_2_vec = {}
    for word in word_to_idx.keys():
        # 词向量矩阵中某个词的索引所对应的那一列即为所该词的词向量
        word_2_vec[word] = wordvec[word_to_idx[word], :]
    save_to = os.path.join(config.output_dir, config.word_to_vec_dict_file)
    with open(save_to, 'wb') as f:
        pickle.dump(word_2_vec, f)

    print(f">>> word_vector_dict saved in {save_to}")

