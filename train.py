from ultralytics import YOLO
import os
import cv2

def train_model():
    model = YOLO('yolov8m-seg.pt')
    model.train(
        data="data.yaml",
        epochs=500,
        imgsz=1024,
        patience=10,
        batch=4,
#        resume=True,
	device=0,
#	workers=os.cpu_count()
    )

if __name__ == "__main__":
    train_model()
