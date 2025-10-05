from ultralytics import YOLO
import os

# 获取当前运行文件的名称（不包括扩展名）
script_name = os.path.splitext(os.path.basename(__file__))[0]

if __name__ == '__main__':
    # Load a model
    model = YOLO(r'/home/yw/data/lxs/xiugai/35gs+dy_youhua/yolo11s-GSconv+ns_ghostconv+dy_ducalcu.yaml')

    # Train the model
    train_results = model.train(
        data=r"/home/yw/data/lxs/xiugai/1dataset/SIMD3junheng.yaml",  # path to dataset YAML
        epochs=200,  # number of training epochs
        imgsz=640,  # training image size
        device="0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
        batch=8,
        patience=50,
        pretrained=False,
        project='/home/yw/data/lxs/xiugai/runs/dy+gs_xiugai',
        name=f'{script_name}'  # 使用运行文件的名称作为目录名
    )
