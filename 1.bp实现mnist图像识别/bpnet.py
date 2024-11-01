import numpy as np

# 定义一个神经网络类 BPNet
class BPNet:
    # 初始化神经网络结构和学习率
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size           # 输入层大小
        self.hidden_size = hidden_size         # 隐藏层大小
        self.output_size = output_size         # 输出层大小
        self.learning_rate = learning_rate     # 学习率

        # 初始化权重和偏置
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1  # 输入层到隐藏层权重
        self.b1 = np.zeros((1, hidden_size))                       # 隐藏层偏置
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1  # 隐藏层到输出层权重
        self.b2 = np.zeros((1, output_size))                       # 输出层偏置

    # 定义 sigmoid 激活函数
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # 计算 sigmoid 的导数
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # 前向传播，计算网络输出
    def forward_propagation(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1    # 隐藏层输入
        self.A1 = self.sigmoid(self.Z1)           # 隐藏层输出
        self.Z2 = np.dot(self.A1, self.W2) + self.b2  # 输出层输入
        self.A2 = self.sigmoid(self.Z2)           # 输出层输出
        return self.A2

    # 反向传播，计算误差并更新权重和偏置
    def backward_propagation(self, X, Y):
        dZ2 = self.A2 - Y                        # 输出层误差
        dW2 = np.dot(self.A1.T, dZ2)             # 输出层权重梯度
        db2 = np.sum(dZ2, axis=0, keepdims=True) # 输出层偏置梯度

        dA1 = np.dot(dZ2, self.W2.T)             # 隐藏层误差传递
        dZ1 = dA1 * self.sigmoid_derivative(self.A1)  # 隐藏层误差
        dW1 = np.dot(X.T, dZ1)                   # 隐藏层权重梯度
        db1 = np.sum(dZ1, axis=0)                # 隐藏层偏置梯度

        # 更新权重和偏置
        self.sgd_update(dW1, db1, dW2, db2)

    # 梯度下降法（SGD） 更新权重和偏置
    def sgd_update(self, dW1, db1, dW2, db2):
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    # 训练网络
    def train(self, X, Y, epochs):
        for epoch in range(epochs):
            self.forward_propagation(X)                # 前向传播
            loss = np.mean(np.square(Y - self.A2))     # 计算损失
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}")
            self.backward_propagation(X, Y)            # 反向传播并更新
        self.save_weights("model_weights.npz")         # 训练后保存权重

    # 保存权重到文件
    def save_weights(self, file_path):
        weights = {'W1': self.W1, 'b1': self.b1, 'W2': self.W2, 'b2': self.b2}
        np.savez(file_path, **weights)
        print(f"Weights saved to {file_path}")

    # 从文件加载权重
    def load_weights(self, file_path):
        with np.load(file_path) as weights:
            self.W1 = weights['W1']
            self.b1 = weights['b1']
            self.W2 = weights['W2']
            self.b2 = weights['b2']
        print(f"Weights loaded from {file_path}")

    # 预测新数据
    def predict(self, X):
        predictions = []
        for x in X:
            A2 = self.forward_propagation(x.reshape(1, -1))  # 前向传播获得输出
            predictions.append(np.argmax(A2, axis=1)[0])     # 取输出最大值对应的类
        return np.array(predictions)
