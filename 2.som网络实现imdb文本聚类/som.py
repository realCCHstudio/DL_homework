import numpy as np


class SOM:
    def __init__(self, m, n, dim, lr=0.5, sigma=2):
        """
        初始化自组织映射（SOM）网络。

        参数：
        m (int): SOM 网格的行数。
        n (int): SOM 网格的列数。
        dim (int): 输入向量的维度。
        lr (float): 初始学习率，决定了权重更新的步幅。
        sigma (float): 初始邻域半径，影响更新邻近神经元的权重。
        """
        self.m = m  # SOM 网格的行数
        self.n = n  # SOM 网格的列数
        self.dim = dim  # 输入向量的维度
        self.lr = lr  # 初始学习率
        self.sigma = sigma  # 初始邻域半径
        self.weights = np.random.rand(m, n, dim)  # 初始化权重矩阵

    def _find_bmu(self, x):
        """
        查找给定输入向量 x 最近的神经元（最佳匹配单元，BMU）。

        参数：
        x (ndarray): 输入向量。

        返回：
        tuple: BMU 的坐标（行，列）。
        """
        distances = np.sqrt(((self.weights - x) ** 2).sum(axis=2))  # 计算每个神经元与输入向量的距离
        return np.unravel_index(distances.argmin(), distances.shape)  # 返回距离最小的神经元坐标

    def _update_weights(self, x, bmu, lr, sigma):
        """
        更新权重。

        参数：
        x (ndarray): 输入向量。
        bmu (tuple): 最佳匹配单元的坐标。
        lr (float): 当前学习率。
        sigma (float): 当前邻域半径。
        """
        for i in range(self.m):
            for j in range(self.n):
                # 计算当前神经元到BMU的距离
                dist = np.sqrt((i - bmu[0]) ** 2 + (j - bmu[1]) ** 2)
                # 只更新在邻域内的节点
                if dist < sigma:
                    influence = np.exp(-dist ** 2 / (2 * sigma ** 2))  # 计算影响力
                    self.weights[i, j] += influence * lr * (x - self.weights[i, j])  # 更新权重

    def train(self, data, num_iterations=1000):
        """
        训练SOM网络。

        参数：
        data (ndarray): 训练数据集，形状为 (样本数, 特征数)。
        num_iterations (int): 训练迭代次数。
        """
        initial_weights = self.weights.copy()  # 保留初始权重，用于计算总变化
        for i in range(num_iterations):
            # 动态调整学习率和邻域半径
            lr = self.lr * (1 - i / num_iterations)
            sigma = self.sigma * (1 - i / num_iterations)

            # 随机选择一个输入向量
            x = data[np.random.randint(0, data.shape[0])]
            # 找到该输入的BMU
            bmu = self._find_bmu(x)
            # 更新权重
            self._update_weights(x, bmu, lr, sigma)

            # 每隔25次迭代打印一次进度
            if i % 25 == 0:
                weight_change = np.sum(np.abs(self.weights - initial_weights))
                print(f"迭代次数 {i}/{num_iterations} - 训练中... 总计权重变化: {weight_change:.4f}")

        print("训练完成。")

    def map_vects(self, data):
        """
        将输入向量映射到SOM网格。

        参数：
        data (ndarray): 输入向量数据集。

        返回：
        list: 每个输入向量对应的BMU坐标列表。
        """
        mapped = []
        for x in data:
            mapped.append(self._find_bmu(x))  # 为每个输入向量找到BMU
        return mapped
