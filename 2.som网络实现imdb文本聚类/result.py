import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF，用于文本特征提取
from sklearn.preprocessing import normalize  # normalize，用于特征向量归一化
from nltk.corpus import stopwords # 自然语言工具包，用于剔除停用词
from nltk.tokenize import word_tokenize # 自然语言工具包，用于文本分割
from som import SOM
from scipy.stats import gaussian_kde

# import nltk
#
# # 下载停用词列表和punkt数据包
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('punkt_tab')

# 设置模型的各项参数
LEARNING_RATE = 0.2
SIGMA = 20
ITERATIONS = 10
SOM_X = 20
SOM_Y = 20
MAX_FEATURES = 40

# 获取英文停用词，并添加自定义停用词
stop_words = set(stopwords.words('english'))
additional_stopwords = {'!', ',', '.', '?', '-s', '-ly', '</s>', 's'}
stop_words.update(additional_stopwords)

#加载数据集以及数据预处理
def load_and_preprocess_data(base_path):
    # 初始化空列表，用于存储处理后的文本数据和标签
    data, labels = [], []

    # 遍历每个类别文件夹 ('pos'、'neg'、'unsup')
    for label in ['pos', 'neg', 'unsup']:
        # 构建当前类别文件夹的路径
        folder_path = os.path.join(base_path, label)

        # 遍历类别文件夹中的所有文件（影评）
        for file_name in os.listdir(folder_path):
            # 读取并打开文件，将文本转为小写
            with open(os.path.join(folder_path, file_name), 'r', encoding='utf-8') as f:
                text = f.read().lower()
                # 使用 word_tokenize 进行分词
                tokens = word_tokenize(text)
                # 过滤掉非字母的词（标点等）和停用词
                tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
                # 将处理后的词汇列表转换为字符串并加入 data 列表
                data.append(' '.join(tokens))
                # 将当前文件的标签加入 labels 列表
                labels.append(label)
    # 返回处理后的文本数据和对应的标签
    return data, labels


# 使用 'train' 文件夹中的数据
data, labels = load_and_preprocess_data('train')

# 特征提取
vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
tfidf_matrix = vectorizer.fit_transform(data)
tfidf_matrix = normalize(tfidf_matrix.toarray())

# 训练SOM
som = SOM(SOM_X, SOM_Y, tfidf_matrix.shape[1], lr=LEARNING_RATE, sigma=SIGMA)
som.train(tfidf_matrix, num_iterations=ITERATIONS)

# 可视化聚类结果
# 将每个数据点映射到 SOM 上的 BMU 节点
mapped = som.map_vects(tfidf_matrix)

# 使用 PCA 将 SOM 的权重降至二维
weights_reshaped = som.weights.reshape(-1, tfidf_matrix.shape[1])
pca = PCA(n_components=2)
weights_2d = pca.fit_transform(weights_reshaped)

# 提取 BMU 的二维坐标
bmu_2d = [weights_2d[x * som.m + y] for x, y in mapped]

# 计算密度
xy = np.vstack([np.array(bmu_2d)[:, 0], np.array(bmu_2d)[:, 1]])
kde = gaussian_kde(xy)
density = kde(xy)

# 第一个绘制函数：根据标签进行着色的散点图
def plot_bmu_scatter_with_labels(ax, bmu_2d, labels):
    colors = {'pos': 'red', 'neg': 'blue', 'unsup': 'green'}

    # 获取所有点的索引并随机打乱
    indices = np.arange(len(labels))
    np.random.shuffle(indices)  # 随机打乱索引

    # 跟踪已经绘制过的标签，避免重复图例
    drawn_labels = set()

    # 根据随机打乱后的索引绘制散点
    for i in indices:
        label = labels[i]
        point = bmu_2d[i]
        if label not in drawn_labels:  # 仅第一次绘制时添加标签
            ax.scatter(point[0], point[1], c=colors[label], alpha=0.5, edgecolors='k', label=label)
            drawn_labels.add(label)
        else:
            ax.scatter(point[0], point[1], c=colors[label], alpha=0.5, edgecolors='k')

# 第二个绘制函数：根据密度进行着色的散点图
def plot_bmu_scatter_with_density(ax, bmu_2d, density):
    scatter = ax.scatter(np.array(bmu_2d)[:, 0], np.array(bmu_2d)[:, 1], c=density, cmap='coolwarm', alpha=0.5)
    return scatter

# 主函数：绘制图形
def plot_all(bmu_2d, labels, density, lr, sigma, iterations):
    plt.figure(1)
    ax1 = plt.subplot(1, 1, 1)
    plt.figure(2)
    ax2 = plt.subplot(1, 1, 1)

    # 绘制根据标签着色的散点图
    plot_bmu_scatter_with_labels(ax1, bmu_2d, labels)
    ax1.set_title(f"PCA Projection of BMU Locations with Labels(Lr = {lr}, Sigma = {sigma}, Iterations = {iterations})")
    ax1.set_xlabel("PCA Component 1")
    ax1.set_ylabel("PCA Component 2")
    ax1.legend()
    ax1.grid(False)
    # 绘制根据密度着色的散点图
    scatter = plot_bmu_scatter_with_density(ax2, bmu_2d, density)
    ax2.set_title(f"PCA Projection of BMU Locations with Density(Lr = {lr}, Sigma = {sigma}, Iterations = {iterations})")
    ax2.set_xlabel("PCA Component 1")
    ax2.set_ylabel("PCA Component 2")
    plt.colorbar(scatter, ax=ax2, label='Density')
    ax2.grid(False)

    plt.show()

# 调用主函数，展示两个图的结合
plot_all(bmu_2d, labels, density,  LEARNING_RATE, SIGMA, ITERATIONS)
print("图表《带标签的 BMU 位置的 PCA 投影》和《具有密度的 BMU 位置的 PCA 投影》绘制完成。")
