from ultralytics import YOLO
import os


script_name = os.path.splitext(os.path.basename(__file__))[0]

if __name__ == '__main__':
    # Load a model
    model = YOLO(r'yolo11s-GSconv+ns_ghostconv+dy_ducalcu.yaml')

    # Train the model
    train_results = model.train(
        data=r"SIMD.yaml",  # path to dataset YAML
        epochs=200,  # number of training epochs
        imgsz=640,  # training image size
        device="0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
        batch=8,
        patience=50,
        pretrained=False,
        project='runs/FOCAL-Ciou+Gsghost+DYAlign',
        name=f'{script_name}'  
    )


