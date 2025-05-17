from flask import Flask, request, jsonify, render_template
import logging
from flask_socketio import SocketIO
from PIL import Image, ImageDraw, ImageFont
import torch
from ultralytics import YOLO
import time
import json
import modules.turret as turret
import modules.get_enemy_pos as get_enemy_pos
import modules.get_obstacles as get_obstacles
import math
import os

app = Flask(__name__)

DEBUG = True
STATE_DEBUG = False

# YOLO 모델 로드
try:
    model = YOLO('best.pt')

except Exception as e:
    raise RuntimeError(f"YOLO model loading failed: {str(e)}")


EXCLUDE_PATHS = ("/info", "/start", "/update_position", "/get_move", "/get_action")
class PathFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return not any(path in msg for path in EXCLUDE_PATHS)
log = logging.getLogger("werkzeug")
log.addFilter(PathFilter())

socketio = SocketIO(app)

# 전역 변수
move_command = []
action_command = []
player_data = {'pos': {'x': 60, 'y': 10, 'z': 57}}  # 기본 위치 설정
destination = {}
impact_info = {}
obstacles = []  # /update_obstacle 데이터 저장
obstacles_from_map = []
latest_nearest_enemy = None
enemy_list = []
dead_enemy_list = []
MATCH_THRESHOLD = 3.0
destination = None
current_position = None
last_position = None
last_valid_angle = None
state = "IDLE"                 # FSM 상태
distance_to_destination = float("inf")
rotation_start_time = None
pause_start_time = None
last_body_x = last_body_y = last_body_z = None

# /info 에서 계산된 최근 제어값
last_control = "STOP"
last_weight  = 0.0

# ── 상수 ──
ROTATION_THRESHOLD_DEG = 1    # 회전 완료 기준 (°)
STOP_DISTANCE = 45.0          # 정지 거리 (m)
SLOWDOWN_DISTANCE = 100.0     # 감속 시작 거리 (m)
ROTATION_TIMEOUT = 0.8        # 회전 최대 시간 (s)
PAUSE_DURATION = 0.5          # 회전 후 일시정지 (s)
WEIGHT_LEVELS = [0.8, 0.6, 0.3, 0.1, 0.05, 0.01]

@app.route('/dashboard')
def dashboard():
    if DEBUG: print('?? dashboard >>>')
    return render_template('dashboard.html')

MOVING = 'PAUSE'
TURRET_FIRST_ROTATING = True
TURRET_HIT = -1

def select_weight(value, levels=WEIGHT_LEVELS):
    return min(levels, key=lambda x: abs(x - value))

def calculate_move_weight(distance):
    if distance <= STOP_DISTANCE:
        return 0.0
    if distance > SLOWDOWN_DISTANCE:
        return 1.0
    norm = (distance - STOP_DISTANCE) / (SLOWDOWN_DISTANCE - STOP_DISTANCE)
    target = 0.01 + (1.0 - 0.01) * (norm ** 2)
    return select_weight(target)

def calculate_rotation_weight(angle_deg):
    if abs(angle_deg) < ROTATION_THRESHOLD_DEG:
        return 0.0
    target = min(0.3, (abs(angle_deg) / 45) * 0.3)
    return select_weight(target)

@app.route('/detect', methods=['POST'])
def detect():
    global player_data, latest_nearest_enemy, action_command, destination, obstacles_from_map, enemy_list
    global TURRET_FIRST_ROTATING, TURRET_HIT, STATE
    print('🌍 detect >>>')

    # 1. 이미지 수신
    image = request.files.get('image')
    if not image:
        return jsonify({"error": "No image received"}), 400

    image_path = 'temp_image.jpg'
    try:
        image.save(image_path)
    except Exception as e:
        return jsonify([])

    # 2. YOLO 탐지
    results = model(image_path, imgsz=640)
    detections = results[0].boxes.data.cpu().numpy()

    # 3. 탐지 결과 필터링
    target_classes = {
        0: 'Car002', 1: 'Car003', 2: 'Car005', 3: 'Human001',
        4: 'Rock001', 5: 'Rock2', 6: 'Tank001', 7: 'Wall001', 8: 'Wall002'
    }
    class_colors = {
        'car002': '#FF0000', 'car003': '#0000FF', 'car005': '#00FF00', 'human001': 'orange',
        'rock001': 'purple', 'rock2': 'yellow', 'tank': '#333388', 'wall001': 'pink', 'wall002': 'brown'
    }

    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", size=20)
    except:
        font = ImageFont.load_default()

    print("Player position:", player_data['pos'])
    
    filtered_results = []
    for index, box in enumerate(detections):
        class_id = int(box[5])
        if class_id not in target_classes:
            continue

        target_name = target_classes[class_id]
        bbox_yolo = [float(item) for item in box[:4]]
        confidence = float(box[4])

        filtered_results.append({
            'id': index,
            'className': target_name,
            'bbox': bbox_yolo,
            'confidence': confidence,
            # 'map_center': coords['map_center'],
            'color': '#0000FF',
            'filled': False,
            'updateBoxWhileMoving': False
        })

    nearest_enemy = {'state': False}
    if STATE == 'PAUSE':
        enemy_list = get_enemy_pos.get_enemy_list(filtered_results, player_data, obstacles_from_map)
        print('📀 nearest_enemy', enemy_list)
    if len(enemy_list) > 0 and STATE == 'PAUSE':
        print('len(enemy_list) > 0')
        try:
            STATE = 'TURRET_ROTATING'
            # if DEBUG: print(f"👉 Generating action command: player_pos={player_data.get('pos')}, dest={destination}")
            latest_nearest_enemy = enemy_list.pop(0)
            action_command = turret.get_action_command(
                player_data['pos'],
                latest_nearest_enemy,
                turret_x_angle=player_data['turret_x'],
                turret_y_angle=player_data['turret_y'],
                player_y_angle=player_data['body_y']
            )
            print('?? action_command', action_command)
        except ValueError as e:
            print(f"?? Error generating action command: {str(e)}")
            action_command = []

    return jsonify(filtered_results)

@app.route('/info', methods=['POST'])
def info():
    global state, destination, current_position, last_position, distance_to_destination
    global rotation_start_time, pause_start_time, last_valid_angle,player_data
    global last_body_x, last_body_y, last_body_z, last_control, last_weight

    data = request.get_json(force=True)
    if not data:
        return jsonify({'error': 'No JSON received'}), 400
    
    # 목적지가 설정되지 않았다면 정지
    if not destination:
        state = "IDLE"
        last_control, last_weight = "STOP", 0.0
        return jsonify(status="success", control="STOP", weight=0.0)

    # 1) 입력 파싱
    p = data.get('playerPos', {})
    bodyX = data.get('playerBodyX', 0.0)
    bodyY = data.get('playerBodyY', 0.0)
    bodyZ = data.get('playerBodyZ', 0.0)
    distance_to_destination = data.get('distance', float("inf"))
    current_position = (p.get('x', 0.0), p.get('z', 0.0))
    player_data = {
        'pos': {
            'x': data.get('playerPos', {}).get('x'),
            'y': data.get('playerPos', {}).get('y'),
            'z': data.get('playerPos', {}).get('z'),
        },
        'turret_x': data.get('playerTurretX'),
        'turret_y': data.get('playerTurretY'),
        'body_x': data.get('playerBodyX'),
        'body_y': data.get('playerBodyY'),
        'body_z': data.get('playerBodyZ'),
    }

@app.route('/update_position', methods=['POST'])
def update_position():
    global player_data
    if DEBUG: print('🚨 update_position >>>')
    data = request.get_json()
    if not data or "position" not in data:
        if DEBUG: print("🚫 Missing position data")
        return jsonify({"status": "ERROR", "message": "Missing position data"}), 400

    try:
        x, y, z = map(float, data["position"].split(","))
        player_data['pos'] = {'x': x, 'y': y, 'z': z}
        player_data.setdefault('turret_x', 0)
        player_data.setdefault('turret_y', 0)
        player_data.setdefault('body_x', 0)
        player_data.setdefault('body_y', 0)
        player_data.setdefault('body_z', 0)
        if DEBUG: print(f"📍 Position updated: {player_data['pos']}")
        return jsonify({"status": "OK", "current_position": player_data['pos']})
    except Exception as e:
        if DEBUG: print(f"🚫 Invalid position format: {str(e)}")
        return jsonify({"status": "ERROR", "message": str(e)}), 400

@app.route('/get_move', methods=['GET'])
def get_move():
    if DEBUG: print('🚨 get_move >>>')
    global move_command
    if move_command:
        command = move_command.pop(0)
        if DEBUG: print(f"🚗 Move Command: {command}")
        return jsonify(command)
    else:
        return jsonify({"move": "STOP", "weight": 1.0})

@app.route('/get_action', methods=['GET'])
def get_action():
    global TURRET_FIRST_ROTATING, TURRET_HIT, STATE
    global action_command, latest_nearest_enemy
    if DEBUG: print('🚨 get_action >>>', action_command)
    if action_command:
        # TURRET_FIRST_ROTATING = False
        command = action_command.pop(0)
        if DEBUG: print(f"?? Action Command: {command}")
        
        if TURRET_HIT == 1 and command['turret'] != 'FIRE' and command['weight'] == 0.0:
            # reverse 끝나는 지점
            # TURRET_FIRST_ROTATING = True
            TURRET_HIT = -1
            STATE = 'PAUSE'
            # print("impact_control False", action_command)

        return jsonify(command)
    else:
        return jsonify({"turret": "", "weight": 0.0})

# 재조준 횟수 저장(3회까지지) -> 변수명 수정 필요요
adjustments_counts = 3
@app.route('/update_bullet', methods=['POST'])
def update_bullet():
    global destination, impact_info, player_data, action_command, latest_nearest_enemy, enemy_list, adjustments_counts, dead_enemy_list
    global TURRET_HIT
    if DEBUG: print('🚨 update_bullet >>>')
    data = request.get_json()
    action_command = []
    if not data:
        if DEBUG: print("🚫 Invalid bullet data")
        return jsonify({"status": "ERROR", "message": "Invalid request data"}), 400

    print(f"💥 Bullet Impact at X={data.get('x')}, Y={data.get('y')}, Z={data.get('z')}, Target={data.get('hit')}")
    impact_info = {
        'x': data.get('x'),
        'y': data.get('y'),
        'z': data.get('z'),
        'target': data.get('hit'),
        'hit': None,
        'timestamp': time.strftime('%H:%M:%S'),
        'tx': latest_nearest_enemy.get('x') if latest_nearest_enemy else None,
        'ty': latest_nearest_enemy.get('y') if latest_nearest_enemy else None,
        'tz': latest_nearest_enemy.get('z') if latest_nearest_enemy else None
    }

    is_hit = turret.is_hit(latest_nearest_enemy, impact_info)
    if DEBUG: print('💥', is_hit)
    if not is_hit:
        TURRET_HIT = 0
    else:
        TURRET_HIT = 1
        dead_enemy_list.append(latest_nearest_enemy['id'])

    if TURRET_HIT == 0 and adjustments_counts != 0:
        # 재조준: 이전에 명중을 못 했고 / 재조준 시도 횟수가 남은 경우
        time.sleep(5)
        adjustments_counts-=1
        try:
            action_command = turret.get_action_command(player_data['pos'], latest_nearest_enemy, impact_info)
            if DEBUG: print('💥 is_hit >> action_command:', action_command)
        except ValueError as e:
            if DEBUG: print(f"🚫 Error generating action command: {str(e)}")
            action_command = []
        
    if TURRET_HIT == 1 or adjustments_counts == 0:
        # 적 리스트의 다음 적 포격: 이전에 명중했거나 / 재조준 시도 횟수가 남지 않은 경우
        if DEBUG: print("💥 Hit!!!!!")
        if len(enemy_list) > 0:
            # 적 리스트 남아있으면 다음 가까운 적 포격
            # turret.get_action_command
            # state는 계속 turret_rotating
            latest_nearest_enemy = enemy_list.pop(0)
            # print('latest_nearest_enemy', latest_nearest_enemy)
            print('🤢', player_data['pos'])
            print('🤢', player_data['turret_x'])
            print('🤢', player_data['turret_y'])
            print('🤢', player_data['body_y'])
            action_command = turret.get_action_command(
                player_data['pos'],
                latest_nearest_enemy,
                turret_x_angle=player_data['turret_x'],
                turret_y_angle=player_data['turret_y'],
                player_y_angle=player_data['body_y']
            )
        
            # print('📀📀 new enemy action_command', action_command)
        else:
            # 적 리스트 남아있지 않은 경우 포신 원위치
            action_command = turret.get_reverse_action_command(
                player_data.get('turret_x', 0),
                player_data.get('turret_y', 0),
                player_data.get('body_x', 0),
                player_data.get('body_y', 0),
            )
            
            print('📀📀 reverse action_command', action_command)

    socketio.emit('bullet_impact', impact_info)
    return jsonify({"status": "OK", "message": "Bullet impact data received"})

@app.route('/set_destination', methods=['POST'])
def set_destination():
    global destination, action_command, state, rotation_start_time, last_position, last_valid_angle

    data = request.get_json()
    action_command = []
    if not data or "destination" not in data:
        if DEBUG: print("?? Missing destination data")
        return jsonify({"status": "ERROR", "message": "Missing destination data"}), 400
    
    try:
        x, y, z = map(float, data["destination"].split(","))
        destination = {'x': x, 'y': y, 'z': z}  # 딕셔너리 형태로 저장

        # 초기 방향 보정을 위해 리셋
        last_position = None
        last_valid_angle = None

        state = "ROTATING"
        rotation_start_time = time.time()
        print(f"?? New destination: {x},{y},{z} (reset last_position)")
        return jsonify(status="OK", destination=destination)
    except Exception as e:
        if DEBUG: print(f"?? Invalid destination format: {str(e)}")
        return jsonify({"status": "ERROR", "message": f"Invalid format: {str(e)}"}), 400
    
@app.route('/update_obstacle', methods=['POST'])
def update_obstacle():
    global obstacles
    if DEBUG: print('🚨 update_obstacle >>>')
    data = request.get_json()
    if not data or 'obstacles' not in data:
        if DEBUG: print("🚫 No obstacle data received")
        logging.warning("No obstacle data received")
        return jsonify({'status': 'error', 'message': 'No data received'}), 400

    obstacles = data['obstacles']
    print(f"🪨 Obstacle data updated:")
    # logging.debug(f"Obstacle data updated: {json.dumps(obstacles, indent=2)}")
    # if DEBUG: print(f"🪨 Obstacle data: {json.dumps(obstacles, indent=2)}")
    return jsonify({'status': 'success', 'message': 'Obstacle data received', 'obstacles_count': len(obstacles)})

@app.route('/init', methods=['GET'])
def init():
    global obstacles_from_map
    global TURRET_FIRST_ROTATING, TURRET_HIT, STATE
    if DEBUG: print('🚨 init >>>')

    config = {
        "startMode": "start",
        "blStartX": 70,
        "blStartY": 10,
        "blStartZ": 45,
        "rdStartX": 60,
        "rdStartY": 10,
        "rdStartZ": 280,
        "detectMode": False,
        "trackingMode": False,
        "logMode": False,
        "enemyTracking": False,
        "saveSnapshot": False,
        "saveLog": False,
        "saveLidarData": False,
        "lux": 30000
    }

    TURRET_FIRST_ROTATING = True
    TURRET_HIT = -1
    STATE = 'PAUSE'
    
    map_path = 'client/NewMap.map'
    obstacles_from_map = get_obstacles.load_obstacles_from_map(map_path)

    if DEBUG: print(f"🛠️ Initialization config sent via /init: {config}")
    return jsonify(config)

@app.route('/start', methods=['GET'])
def start():
    global obstacles_from_map
    if DEBUG: print("🚀 /start command received")
    map_path = 'client/NewMap2.map'
    obstacles_from_map = get_obstacles.load_obstacles_from_map(map_path)
    print('obstacles_from_map', obstacles_from_map)
    return jsonify({"control": ""})

@app.route('/test_rotation', methods=['POST'])
def test_rotation():
    global action_command
    if DEBUG: print('🚨 test_rotation >>>')
    data = request.get_json()
    rotation_type = data.get('type', 'Q')
    count = data.get('count', 1)

    action_command = []
    for _ in range(count):
        action_command.append({"turret": rotation_type, "weight": 0.5})
    action_command.append({"turret": rotation_type, "weight": 0.0})

    test_info = {
        'rotation_type': rotation_type,
        'count': count,
        'timestamp': time.strftime('%H:%M:%S'),
        'rotation_desc': {
            'Q': 'Left', 'E': 'Right', 'F': 'Down', 'R': 'Up'
        }.get(rotation_type, 'Unknown')
    }
    if DEBUG: print(f"🔄 Testing {test_info['rotation_desc']} rotation ({rotation_type}) x {count}")
    socketio.emit('rotation_test', test_info)
    if DEBUG: print("action_command >>", action_command)
    return jsonify({"status": "OK", "message": "Rotation test started"})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)