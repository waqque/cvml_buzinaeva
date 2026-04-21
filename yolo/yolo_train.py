from ultralytics import YOLO
from pathlib import Path
import yaml

def train_model():

    class_labels = {0: "cube", 1: "neither", 2: "sphere"}

    data_dir = Path("./spheres_and_cubes_new").resolve()

    dataset_config = {
        "path": str(data_dir.absolute()),
        "train": str((data_dir / "images/train").absolute()),
        "val": str((data_dir / "images/val").absolute()),
        "nc": len(class_labels),
        "names": class_labels
    }
    
    with open(data_dir / "dataset.yaml", "w", encoding="utf-8") as f:
        yaml.dump(dataset_config, f, allow_unicode=True)

    model_size = "n"  # n, s, m, l, x
    model = YOLO(f"yolo26{model_size}.pt")
    

    results = model.train(
        data=data_dir / "dataset.yaml",
        epochs=50,
        batch=8,
        workers=0,              
        device="cpu",
        patience=5,
        optimizer="AdamW",
        lr0=0.001,
        

        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        degrees=5.0,
        scale=0.5,
        translate=0.1,
        
        conf=0.001,
        iou=0.7,
        
        project=f"{data_dir}/figures",
        name="yolo11_cpu",
        save=True,
        save_period=5,
        verbose=True,
        plots=True,
        val=True,
        close_mosaic=8,
        amp=False,
        warmup_epochs=5,
        cos_lr=True,
        dropout=0.2,
        imgsz=416
    )
    
    print(f"Модель сохранена: {results.save_dir / 'weights' / 'best.pt'}")
    return results.save_dir / "weights" / "best.pt"

if __name__ == "__main__":
    train_model()