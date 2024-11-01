import os
import numpy as np
from PIL import Image
from bpnet import BPNet

# 定义读取图像的函数，将图像转换为28x28像素并扁平化
def load_test_images(folder, size=(28, 28), threshold=0.5):
    images = []
    labels = []
    filenames = []
    for filename in os.listdir(folder):
        if filename.endswith('.jpg'):
            img = Image.open(os.path.join(folder, filename)).convert('L')  # 灰度化
            img = img.resize(size)  # 调整为 28x28 尺寸
            img_array = np.array(img) / 255.0  # 归一化

            # 二值化处理
            img_array = (img_array > threshold).astype(float)  # 大于阈值的设为1，其余为0

            images.append(img_array.flatten())  # 扁平化
            filenames.append(filename)  # 记录文件名

            # 从文件名中提取真实标签值，例如 "test_0_7.jpg" 中的 "7"
            label = int(filename.split('_')[-1].split('.')[0])
            labels.append(label)
    return np.array(images), labels, filenames

# 设置网络参数
input_size = 28 * 28
hidden_size = 128
output_size = 10

# 创建 BPNet 实例并加载权重
bp_net = BPNet(input_size, hidden_size, output_size)
bp_net.load_weights("model_weights.npz")

# 加载测试数据
test_images, true_labels, filenames = load_test_images('test')

# 获取预测结果
predictions = bp_net.predict(test_images)

# 输出每张测试图片的预测结果并计算准确率
correct_count = 0
for i, prediction in enumerate(predictions):
    print(f"图片: {filenames[i]}, 预测值: {prediction}, 真实值: {true_labels[i]}")
    if prediction == true_labels[i]:
        correct_count += 1

# 计算并输出总体准确率
accuracy = correct_count / len(true_labels)
print(f"总体准确率: {accuracy:.3%}")
