from ultralytics import YOLO
import torch
import gc

model = YOLO("uxo_detection/models/best_1280.onnx")

#model.export(format="onnx",imgsz=1280,quantize=16)

for i, result in enumerate(model.predict(
    source="video/test.mkv",
    #project="runs/detect/1280_onnx",
    imgsz=640,
    conf=0.7,
    iou=0.45,
    stream=True,
    device=0,
    save=True,
    show=False
)):
    del result

    if i % 500 == 0:
        gc.collect()
        torch.cuda.empty_cache()

        print(
            f"Frame: {i}, "
            f"GPU allocated: "
            f"{torch.cuda.memory_allocated() / 1024**2:.1f} MB, "
            f"GPU reserved: "
            f"{torch.cuda.memory_reserved() / 1024**2:.1f} MB"
        )