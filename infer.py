# infer.py (B 컨테이너 안에 있는 파일)
import torch
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # 사전 학습된 모델 사용

def run_inference(image_path):
    results = model(image_path)
    return results[0].boxes.xyxy  # 예: 바운딩 박스 정보 반환

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("사용법: python infer.py <image_path>")
    else:
        output = run_inference(sys.argv[1])
        print(output)
