#红绿灯
import os
# 禁用全局设置文件
os.environ['YOLO_SETTINGS'] = ''
from ultralytics import YOLO
if __name__ == "__main__":
    # 加载模型
    model = YOLO("ultralytics/cfg/models/custom/cch_yolo.yaml")

    # 训练模型
    train_results = model.train(
        data="datasets/dataset.yaml",  # 数据集 YAML 路径
        epochs=5,  # 训练轮次
        imgsz=1024,  # 训练图像尺寸
        device="0",  # 运行设备，例如 device=0 或 device=0,1,2,3 或 device=cpu
        batch=2
    )