from ultralytics import YOLO
import torch

def main():
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

    DATA_DIR = r"Asset\DatasetSplit"

    # Use small model (best for 900 correlated images)
    model = YOLO("yolo11s-cls.pt")

    model.train(
        data=DATA_DIR,
        epochs=100,        # more epochs
        imgsz=224,
        batch=32,
        lr0=0.001,
        patience=15,       # early stopping
        device=device,
        amp=True,
        augment=True,      # stronger augmentations
        verbose=True
    )

if __name__ == "__main__":
    main()
