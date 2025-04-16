from ultralytics import YOLO
import os
import argparse
import yaml
from pathlib import Path

def train_yolo(args):
    """
    YOLO 모델 학습 및 테스트 함수
    
    Args:
        args: 커맨드 라인 인자
            - data: 데이터 설정 파일 경로
            - epochs: 학습 에포크 수
            - batch: 배치 크기
            - img_size: 이미지 크기
            - device: 사용할 디바이스 (cpu/gpu)
    """
    # 1. 모델 로드
    print("모델 로드 중...")
    model = YOLO('yolov8n.pt')  # 가장 작은 YOLOv8 모델 사용
    print("모델 로드 완료")
    
    # 2. 학습 설정
    print("학습 시작...")
    results = model.train(
        data=args.data,              # 데이터 설정 파일
        epochs=args.epochs,          # 학습 에포크
        imgsz=args.img_size,         # 이미지 크기
        batch=args.batch,            # 배치 크기
        project='runs/train',        # 프로젝트 디렉토리
        name='test_model',           # 모델 저장 이름
        device=args.device,          # 사용할 디바이스
        workers=2,                   # workers 수 줄임
        cache=False,                 # 캐시 비활성화
        patience=50,                 # Early stopping patience
        save=True,                   # 모델 저장
        save_period=10,              # 10 에포크마다 저장
        pretrained=True,             # 사전 학습된 가중치 사용
        optimizer='auto',            # 최적화 알고리즘
        verbose=True,                # 상세한 출력
        seed=42                      # 재현성을 위한 시드 설정
    )
    print("학습 완료")
    
    # 3. 학습된 모델로 예측 테스트
    print("예측 테스트 시작...")
    
    # data.yaml 파일에서 테스트 이미지 경로 읽기
    with open(args.data, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # 기본 경로와 테스트 경로 가져오기
    base_path = data_config.get('path', '/app/test_data')
    test_path = data_config.get('test', 'test/images')
    
    # 테스트 이미지 경로 구성
    test_dir = os.path.join(base_path, test_path)
    
    # 테스트 디렉토리에서 첫 번째 이미지 파일 찾기
    test_images = list(Path(test_dir).glob('*.jpg'))
    if test_images:
        test_image = str(test_images[0])
        print(f"테스트 이미지: {test_image}")
        
        # 예측 임계값을 낮춰서 더 많은 객체를 감지하도록 설정
        results = model.predict(
            test_image, 
            save=True,
            conf=0.25,  # 신뢰도 임계값 낮춤
            iou=0.45,   # IoU 임계값
            project='runs/detect',  # 프로젝트 디렉토리
            name='predict'          # 결과 저장 이름
        )
        print(f"예측 완료! 결과는 runs/detect/predict 폴더에 저장됩니다.")
        
        # 학습 결과 확인
        print("\n학습 결과 확인:")
        print(f"학습된 모델 가중치: runs/train/test_model/weights/best.pt")
        print(f"학습 메트릭스: runs/train/test_model/results.csv")
    else:
        print(f"테스트 이미지를 찾을 수 없습니다: {test_dir}")
        print("테스트 디렉토리에 이미지 파일이 있는지 확인하세요.")

def main():
    parser = argparse.ArgumentParser(description='YOLO 모델 학습 및 테스트')
    parser.add_argument('--data', type=str, default='/app/test_data/data.yaml',
                      help='데이터 설정 파일 경로')
    parser.add_argument('--epochs', type=int, default=50,
                      help='학습 에포크 수')
    parser.add_argument('--batch', type=int, default=2,
                      help='배치 크기')
    parser.add_argument('--img_size', type=int, default=640,
                      help='이미지 크기')
    parser.add_argument('--device', type=str, default='0',
                      help='사용할 디바이스 (cpu 또는 gpu 번호)')
    
    args = parser.parse_args()
    train_yolo(args)

if __name__ == "__main__":
    main()
