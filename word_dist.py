import os
import numpy as np
from lib.config import Config


def top_k_nearest_words(word, k, word2vec, word2id, id2word):
    """
    Args:
        word: 词
        k: 最相似的k个词
        word2vec: 词向量
        word2id: 词与id的对应关系
        id2word: id与词的对应关系
    """

    # 计算词与词之间的余弦相似度
    cos = np.dot(
        word2vec, word2vec[word2id[word]]
    ) / (np.linalg.norm(word2vec, axis=1) * np.linalg.norm(word2vec[word2id[word]]))
    # 按相似度排序
    cos = np.argsort(-cos)
    # 取出最相似的k个词
    k_nearest_words = cos[1:k+1]

    words = [id2word[i] for i in k_nearest_words]
    # 输出最相似的k个词
    for wd in words:
        print("\t" + wd, end="")
    return words


def find_top_k_nearest_words(config, words_lst, top_k):

    # 读取词向量
    load_from = os.path.join(config.output_dir, config.word_to_vec_file)
    word2vec = np.load(load_from, allow_pickle=True)
    # 读取词与id的对应关系
    load_from = os.path.join(config.output_dir, config.word_to_idx_file)
    word2id = np.load(load_from, allow_pickle=True)
    # 读取id与词的对应关系
    load_from = os.path.join(config.output_dir, config.idx_to_word_file)
    id2word = np.load(load_from, allow_pickle=True)

    save_to = os.path.join(config.output_dir, f"top_{top_k}_nearest_words.txt")
    for word in words_lst:
        # 输出与某个词最相似的top_k个词
        str_info = f"与[{word}]最相似的{top_k}个词为："
        print(str_info)
        words = top_k_nearest_words(word, top_k, word2vec, word2id, id2word)
        print("\n")
        # 保存结果
        with open(save_to, "a") as f:
            f.write(str_info + "\n")
            for w in words:
                f.write("\t" + w)
            f.write("\n")
    print(f">>> top_{top_k}_nearest_words saved in {save_to}")


if __name__ == '__main__':

    top_k = 10
    word_lst = ["金融", "扶贫", "信贷", "证券", "保险", "贷款", "贴息", "价格"]
    find_top_k_nearest_words(Config, word_lst, top_k)
