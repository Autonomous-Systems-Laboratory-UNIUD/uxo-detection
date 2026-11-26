from ultralytics import YOLO

model = YOLO("models/best_ncnn_model_320")

results = model(
    source="video/video.mkv",
    conf=0.7,
    iou=0.45,
    save=True,
    device="cpu",
    show=False,
    stream=True
)

for i, result in enumerate(results):
    print(f"Processing frame {i}: {len(result.boxes)} detections")

print("Video processing completed!")
