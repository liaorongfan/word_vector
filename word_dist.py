import numpy as np


def top_k_nearest_words(word, k, word2vec, word2id):
    """
    Args:
        word: 词
        k: 最相似的k个词
        word2vec: 词向量
        word2id: 词与id的对应关系
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


if __name__ == '__main__':
    # 读取词向量
    word2vec = np.load("agriculture_wordvec_cbow.pkl", allow_pickle=True)
    # 读取词与id的对应关系
    word2id = np.load("word2idx.pkl", allow_pickle=True)
    # 读取id与词的对应关系
    id2word = np.load("idx2word.pkl", allow_pickle=True)
    # 计算词与词之间的余弦相似度
    top_k = 10
    word_lst = ["金融", "扶贫", "信贷", "证券", "保险", "贷款", "贴息", "价格"]
    for word in word_lst:
        # 输出与某个词最相似的top_k个词
        str_info = f"与[{word}]最相似的{top_k}个词为："
        print(str_info)
        words = top_k_nearest_words(word, top_k, word2vec, word2id)
        print("\n")
        # 保存结果
        with open(f"top_{top_k}_nearest_words.txt", "w") as f:
            f.write(str_info + "\n")
            for w in words:
                f.write("\t" + w)
            f.write("\n")
