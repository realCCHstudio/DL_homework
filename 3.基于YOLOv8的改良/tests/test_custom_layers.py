import torch
from ultralytics.models import YOLO

# 使用自定义模型配置
model = YOLO('ultralytics/cfg/models/custom/cch_yolo.yaml')

# 创建随机输入数据
x = torch.randn(1, 3, 640, 640)  # 以 640x640 的输入大小为例
y = model(x)

print("模型输出：", y)
