from ultralytics import YOLO

model = YOLO('yolo/yolov8n-pose.pt')
result = model(source="video/IMG_5500.mp4", conf=.5, save=True, show=True, save_txt=True)