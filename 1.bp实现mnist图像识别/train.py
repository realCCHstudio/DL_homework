import os
import numpy as np
from PIL import Image
from bpnet import BPNet

# 定义读取图像的函数，将图像转换为28x28像素并扁平化
# 定义加载并二值化图像的函数
def load_images(folder, size=(28, 28), threshold=0.5):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith('.jpg'):
            label = int(filename.split('_')[-1].split('.')[0])  # 提取标签
            img = Image.open(os.path.join(folder, filename)).convert('L')  # 灰度化
            img = img.resize(size)  # 调整为 28x28 尺寸
            img_array = np.array(img) / 255.0  # 归一化

            # 二值化处理
            img_array = (img_array > threshold).astype(float)  # 大于阈值的设为1，其余为0

            images.append(img_array.flatten())  # 扁平化
            labels.append(label)
    return np.array(images), np.array(labels)

# 设置网络参数
input_size = 28 * 28
hidden_size = 128
output_size = 10
learning_rate = 0.0001
epochs = 1500

# 加载训练数据
train_images, train_labels = load_images('train')

# 将标签转换为One-Hot编码
train_labels_one_hot = np.eye(output_size)[train_labels]

# 创建 BPNet 实例
bp_net = BPNet(input_size, hidden_size, output_size, learning_rate)

# 训练模型并保存权重
bp_net.train(train_images, train_labels_one_hot, epochs)
