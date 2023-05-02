from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("./runs/detect/train2/weights/best.pt")
    # model.train(data='./Datasets/GTSDB/data.yaml', workers=4)
    model.predict(source=0, show=True)
