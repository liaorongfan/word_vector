from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


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
