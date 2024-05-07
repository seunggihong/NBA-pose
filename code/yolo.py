from ultralytics import YOLO

model = YOLO('yolo/yolov8n-pose.pt')
result = model(source="video/test_video.mp4", conf=.3, show=True)