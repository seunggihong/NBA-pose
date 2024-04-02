from ultralytics import YOLO

model = YOLO('yolo/yolov8n-pose.pt')
result = model(source="video/golf.mp4", conf=.5, save=True, show=True, save_txt=True)

print(result[0].boxes)