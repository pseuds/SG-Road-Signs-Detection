from ultralytics import YOLO

# load model
model = YOLO("yolov8s.pt")

results = model.train(
    data="/home/s14-htx/Documents/GitHub/GPU_file_sharing/SG-Road-Signs-2/data.yaml",
    epochs=100, imgsz=800
    )