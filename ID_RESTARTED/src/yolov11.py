
from ultralytics import YOLO

if __name__ == '__main__':
    # Load a pretrained YOLO11n model
    model = YOLO(r"D:\python\ID CARD DETECTION\ID_RESTARTED\models\yolo11n.pt")

    # Start a new training session on GPU
    train_results = model.train(
        data=r"D:\python\ID CARD DETECTION\ID_RESTARTED\config\config.yaml",
        epochs=100,
        device=0,
        workers=0,
        exist_ok=True,
        project=r"D:\python\ID CARD DETECTION\ID_RESTARTED\runs",
        name="detect"
    )

