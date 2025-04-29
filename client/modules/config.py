# config.py

# 상수
ROTATION_THRESHOLD_DEG = 5
STOP_DISTANCE = 45.0
SLOWDOWN_DISTANCE = 120.0
ROTATION_TIMEOUT = 0.8
PAUSE_DURATION = 0.5
WEIGHT_LEVELS = [1.0, 0.6, 0.3, 0.1, 0.05, 0.01]
DETECTION_RANGE = 100.0
SHOOTING_RANGE = 45.0  # 포격 거리 기준
OBSTACLE_BUFFER = 5.0  # 장애물 회피 거리 (미터)

# 클래스 정의
ENEMY_CLASSES = {'car2', 'car3', 'tank'}
FRIENDLY_CLASSES = {'car5'}
OBSTACLE_CLASSES = {'rock1', 'rock2', 'wall1', 'wall2'}

# YOLO 클래스 매핑
TARGET_CLASSES = {
    0: 'car2',
    1: 'car3',
    2: 'car5',
    3: 'human1',
    4: 'rock1',
    5: 'rock2',
    6: 'tank',
    7: 'wall1',
    8: 'wall2'
}