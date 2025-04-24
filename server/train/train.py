from ultralytics import YOLO
import os
import argparse
import yaml
from pathlib import Path

def train_yolo(args):
    """
    YOLO 모델 학습 및 검증 함수
    
    Args:
        args: 커맨드 라인 인자
            - data: 데이터 설정 파일 경로
            - epochs: 학습 에포크 수
            - batch: 배치 크기
            - img_size: 이미지 크기
            - device: 사용할 디바이스 (cpu/gpu)
            - val_ratio: 검증 데이터 비율 (0.0 ~ 1.0)
    """
    # 1. 모델 로드
    print("모델 로드 중...")
    model = YOLO('/app/server/train/runs/train/roboflow_yolov8n/weights/best.pt')
    print("모델 로드 완료")
    
    # data.yaml 파일 읽기
    with open(args.data, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # 2. 학습 설정
    print("\n=== 학습 설정 ===")
    print(f"데이터 경로:")
    print("- 학습 이미지:")
    for path in data_config['train']:
        print(f"  • {path}")
    print("\n- 학습 라벨:")
    for path in data_config['train_labels']:
        print(f"  • {path}")
    print(f"\n검증 데이터 비율: {args.val_ratio * 100}%")
    
    # 데이터셋 준비
    base_path = data_config['path']
    total_images = 0
    
    # 모든 이미지 경로에서 이미지 수 계산
    for img_path in data_config['train']:
        full_path = os.path.join(base_path, img_path)
        total_images += len(list(Path(full_path).glob('*.jpg')))
    
    val_size = int(total_images * args.val_ratio)
    train_size = total_images - val_size
    
    print(f"\n총 이미지 수: {total_images}")
    print(f"- 학습 데이터: {train_size}개")
    print(f"- 검증 데이터: {val_size}개")
    
    print("\n학습 시작...")
    results = model.train(
        data=args.data,              # 데이터 설정 파일
        epochs=args.epochs,          # 학습 에포크
        imgsz=args.img_size,         # 이미지 크기
        batch=args.batch,            # 배치 크기
        project='runs/train',        # 프로젝트 디렉토리
        name='roboflow_yolov8n',     # 모델 저장 이름
        device=args.device,          # 사용할 디바이스
        workers=0,                   # worker 수 감소
        cache=False,                 # 캐시 비활성화
        patience=50,                 # Early stopping patience
        save=True,                   # 모델 저장
        save_period=10,              # 10 에포크마다 저장
        pretrained=True,             # 사전 학습된 가중치 사용
        optimizer='auto',            # 최적화 알고리즘
        verbose=True,                # 상세한 출력
        seed=42,                     # 재현성을 위한 시드 설정
        split=args.val_ratio,        # 검증 데이터 분할 비율
        overlap_mask=False,          # 마스크 오버랩 비활성화
        mask_ratio=4,                # 마스크 비율 감소
        nbs=64,                      # 공칭 배치 크기
        close_mosaic=10,             # 모자이크 조기 비활성화
        amp=False,                   # 자동 혼합 정밀도 비활성화
        fraction=1.0,                # 데이터셋 분율
        exist_ok=True,               # 기존 실험 덮어쓰기 허용
    )
    print("학습 완료")
    
    # 3. 검증 수행
    print("\n모델 검증 중...")
    metrics = model.val(
        data=args.data,          # data.yaml 파일 경로
        split='val',             # 검증 데이터셋 사용
        imgsz=args.img_size,     # 이미지 크기
        batch=16,                # 배치 크기
        conf=0.3,                # 신뢰도 임계값 (0.3이 일반적)
        iou=0.5,                 # IoU 임계값 (0.5가 표준)
        project='runs/detect',   # 프로젝트 디렉토리
        name='val_results',      # 결과 저장 이름
        save_json=True,          # JSON 형식으로 결과 저장
        save_txt=True,           # 텍스트 형식으로 결과 저장
        save_conf=True,          # 신뢰도 점수 저장
        plots=True,              # 성능 그래프 생성
        verbose=True             # 상세한 출력 활성화
    )
    
    # 검증 결과 출력
    print("\n=== 검증 결과 ===")
    print(f"mAP@.5 (IoU=0.5): {metrics.box.map50:.3f}")          # mAP@0.5
    print(f"mAP@.5:.95 (IoU=0.5-0.95): {metrics.box.map:.3f}")  # mAP@0.5:0.95
    print(f"Precision (정밀도): {metrics.box.mp:.3f}")            # Mean Precision
    print(f"Recall (재현율): {metrics.box.mr:.3f}")               # Mean Recall
    print(f"F1-Score: {2 * (metrics.box.mp * metrics.box.mr) / (metrics.box.mp + metrics.box.mr):.3f}")  # F1 Score
    
    # 결과 파일 경로 출력
    print("\n=== 결과 파일 경로 ===")
    print(f"학습된 모델 가중치: runs/train/roboflow_yolov8n/weights/best.pt")
    print(f"학습 메트릭스: runs/train/roboflow_yolov8n/results.csv")
    print(f"검증 결과: runs/detect/val_results")
    print(f"검증 결과 시각화: runs/detect/val_results/val_batch0_pred.jpg")

def main():
    parser = argparse.ArgumentParser(description='YOLO 모델 학습 및 검증')
    parser.add_argument('--data', type=str, default='/app/test_data/data.yaml',
                      help='데이터 설정 파일 경로')
    parser.add_argument('--epochs', type=int, default=50,
                      help='학습 에포크 수')
    parser.add_argument('--batch', type=int, default=4,  # 배치 크기를 1로 감소
                      help='배치 크기')
    parser.add_argument('--img_size', type=int, default=640,
                      help='이미지 크기')
    parser.add_argument('--device', type=str, default='0',
                      help='사용할 디바이스 (cpu 또는 gpu 번호)')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                      help='검증 데이터 비율 (0.0 ~ 1.0)')
    
    args = parser.parse_args()
    train_yolo(args)

if __name__ == "__main__":
    main()
