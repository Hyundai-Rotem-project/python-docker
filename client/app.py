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
MATCH_THRESHOLD = 3.0
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

def iou(box1, box2):
    # box: [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

def nms(detections, iou_threshold=0.5):
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    keep = []
    while detections:
        det = detections.pop(0)
        keep.append(det)
        detections = [d for d in detections if iou(det['bbox'], d['bbox']) < iou_threshold or det['className'] != d['className']]
    return keep

@app.route('/detect', methods=['POST'])
def detect():
    global player_data, latest_nearest_enemy, action_command, destination, obstacles_from_map
    global TURRET_FIRST_ROTATING, TURRET_HIT
    print('?? detect >>>')

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
            'color': '#0000FF',
            'filled': False,
            'updateBoxWhileMoving': False
        })

    if STATE_DEBUG : print('1 ????TURRET_FIRST_ROTATING', TURRET_FIRST_ROTATING)
    if STATE_DEBUG : print('1 ????TURRET_HIT', TURRET_HIT)

    # state가 STOPPED일 때만 action_command 생성
    if state == "STOPPED":
        print("?? Current state: STOPPED, searching for enemies...")
        nearest_enemy = get_enemy_pos.find_nearest_enemy(filtered_results, player_data, obstacles_from_map)
        print('?? nearest_enemy', nearest_enemy)
        if nearest_enemy['state'] and TURRET_FIRST_ROTATING:
            try:
                latest_nearest_enemy = nearest_enemy
                print("?? Generating action command for enemy:", nearest_enemy)
                action_command = turret.get_action_command(
                    player_data['pos'],
                    nearest_enemy,
                    turret_x_angle=player_data['turret_x'],
                    turret_y_angle=player_data['turret_y'],
                    player_y_angle=player_data['body_y']
                )
                print('?? action_command', action_command)
            except ValueError as e:
                print(f"?? Error generating action command: {str(e)}")
                action_command = []
            
            if STATE_DEBUG : print('2 ????action - TURRET_FIRST_ROTATING f', TURRET_FIRST_ROTATING)
            if STATE_DEBUG : print('2 ????action - TURRET_HIT -1', TURRET_HIT)
    else:
        # 이동 중에는 action_command 비우기
        action_command = []

    filtered_results = nms(filtered_results)

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

    # 2) 초기 방향 보정
    if last_position and current_position != last_position:
        dx = current_position[0] - last_position[0]
        dz = current_position[1] - last_position[1]
        if math.hypot(dx, dz) > 1e-4:
            current_angle = math.atan2(dz, dx)
        else:
            current_angle = math.radians(bodyX)
    else:
        dx = destination['x']
        dz = destination['z']
        px, pz = current_position
        px, pz = current_position
        current_angle = math.atan2(dz - pz, dx - px)
    last_valid_angle = current_angle

    # 3) 바디 방향 변화 로그
    if last_body_x is not None:
        dbx, dby, dbz = bodyX - last_body_x, bodyY - last_body_y, bodyZ - last_body_z
        if abs(dbx) < 1e-3 and state == "ROTATING":
            print("?? bodyX change too small during ROTATING")
        print(f"?? Δbody: X={dbx:.3f}, Y={dby:.3f}, Z={dbz:.3f}")
    last_body_x, last_body_y, last_body_z = bodyX, bodyY, bodyZ

    # 4) FSM 처리
    control, weight = "STOP", 0.0

    if state == "IDLE":
        state = "ROTATING"
        rotation_start_time = time.time()

    elif state == "ROTATING":
        dx = destination['x']
        dz = destination['z']
        px, pz = current_position

        # 현재 전방 벡터
        fx, fz = math.cos(current_angle), math.sin(current_angle)
        # 목표 방향 벡터 (정규화)
        tx, tz = dx - px, dz - pz
        dist = math.hypot(tx, tz)
        if dist > 1e-6:
            tx /= dist
            tz /= dist

        # 내적으로 각도 차이 계산
        dot = max(-1.0, min(1.0, fx*tx + fz*tz))
        angle_diff_rad = math.acos(dot)
        deg = math.degrees(angle_diff_rad)
        # 외적(z 성분)으로 회전 방향 판별
        cross = fx * tz - fz * tx

        print(f"?? ROTATING: angle_diff={deg:.2f}°, cross={cross:.3f}")

        # 회전 타임아웃 또는 완료 판정
        if rotation_start_time and (time.time() - rotation_start_time) > ROTATION_TIMEOUT:
            state = "PAUSE"
            pause_start_time = time.time()
        elif deg < ROTATION_THRESHOLD_DEG:
            state = "PAUSE"
            pause_start_time = time.time()
        else:
            control = "A" if cross > 0 else "D"
            weight = calculate_rotation_weight(deg)

    elif state == "PAUSE":
        if (time.time() - pause_start_time) >= PAUSE_DURATION:
            state = "MOVING"
            control = "W"
            weight = calculate_move_weight(distance_to_destination)

    elif state == "MOVING":
        dx = destination['x']
        dz = destination['z']
        px, pz = current_position
        z_diff = abs(pz - dz)

        # 방향 재판단에도 동일한 벡터 로직 사용
        fx, fz = math.cos(current_angle), math.sin(current_angle)
        tx, tz = dx - px, dz - pz
        dist = math.hypot(tx, tz)
        if dist > 1e-6:
            tx /= dist
            tz /= dist
        dot = max(-1.0, min(1.0, fx*tx + fz*tz))
        angle_diff_rad = math.acos(dot)
        deg = math.degrees(angle_diff_rad)
        cross = fx * tz - fz * tx

        # 도착 조건
        if distance_to_destination <= STOP_DISTANCE or z_diff < 20.0:
            state = "STOPPED"
        # 큰 방향 오류 시 재회전
        elif abs(deg) > ROTATION_THRESHOLD_DEG * 6:
            state = "ROTATING"
            rotation_start_time = time.time()
            control = "A" if cross > 0 else "D"
            weight  = calculate_rotation_weight(deg)
        else:
            control = "W"
            weight  = calculate_move_weight(distance_to_destination)

    else:  # STOPPED
        control, weight = "STOP", 0.0

    # 5) 결과 저장 및 반환
    last_control, last_weight = control, weight
    last_position = current_position

    return jsonify(status="success", control=control, weight=weight)

@app.route('/update_position', methods=['POST'])
def update_position():
    global current_position, last_position, state, destination, player_data

    if DEBUG: print('🚨 update_position >>>')

    data = request.get_json()
    if not data or "position" not in data:
        return jsonify({'status': 'ERROR', 'message': 'Missing position data'}), 400
    try:
        x, y, z = map(float, data["position"].split(","))
        player_data['pos'] = {'x': x, 'y': y, 'z': z}
        player_data.setdefault('turret_x', 0)
        player_data.setdefault('turret_y', 0)
        player_data.setdefault('body_x', 0)
        player_data.setdefault('body_y', 0)
        player_data.setdefault('body_z', 0)
        current_position = (x, z)
        if last_position:
            dx, dz = x - last_position[0], z - last_position[1]
            print(f"?? Movement change: dx={dx:.6f}, dz={dz:.6f}")
        if destination:
            dx = destination['x']
            dz = destination['z']
            z_diff = abs(z - dz)
            direction_angle = math.degrees(math.atan2(dz - z, dx - x))
            print(f"?? Position updated: {current_position}, target angle: {direction_angle:.2f}°, z_diff: {z_diff:.2f}m")
        else:
            print(f"?? Position updated: {current_position}")
        return jsonify(status="OK", current_position=current_position)
    except Exception as e:
        return jsonify({'status': 'ERROR', 'message': str(e)}), 400
    
@app.route('/get_move', methods=['GET'])
def get_move():
    # /info에서 계산된 제어값을 그대로 반환
    return jsonify(move=last_control, weight=last_weight)

@app.route('/get_action', methods=['GET'])
def get_action():
    global TURRET_FIRST_ROTATING, TURRET_HIT, MOVING
    global action_command, latest_nearest_enemy
    if DEBUG: print('?? get_action >>>', action_command)
    print("state :"+str(state))
    
    # state가 STOPPED일 때만 action_command 반환
    if state == "STOPPED":
        if action_command:
            TURRET_FIRST_ROTATING = False
            command = action_command.pop(0)
            if DEBUG: print(f"?? Action Command: {command}")
            if TURRET_HIT == 1 and command['turret'] != 'FIRE' and command['weight'] == 0.0:
                TURRET_FIRST_ROTATING = True
                TURRET_HIT = -1
                MOVING = 'MOVING'
                if STATE_DEBUG : print('5 ????reverse end - TURRET_FIRST_ROTATING t', TURRET_FIRST_ROTATING)
                if STATE_DEBUG : print('5 ????reverse end - TURRET_HIT -1', TURRET_HIT)
            return jsonify(command)
        else:
            return jsonify({"turret": "", "weight": 0.0})
    else:
        # 이동 중에는 항상 빈 명령 반환
        return jsonify({"turret": "", "weight": 0.0})
    

@app.route('/update_bullet', methods=['POST'])
def update_bullet():
    global destination, impact_info, player_data, action_command, latest_nearest_enemy, TURRET_HIT
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
        'timestamp': time.strftime('%H:%M:%S')
    }

    is_hit = turret.is_hit(latest_nearest_enemy, impact_info)
    if DEBUG: print('💥', is_hit)
    if not is_hit:
        TURRET_HIT = 0
        time.sleep(5)
        try:
            action_command = turret.get_action_command(player_data['pos'], latest_nearest_enemy, impact_info)
            if DEBUG: print('💥 is_hit >> action_command:', action_command)
        except ValueError as e:
            if DEBUG: print(f"🚫 Error generating action command: {str(e)}")
            action_command = []
        
        if STATE_DEBUG : print('3 🤩🤩re action - TURRET_FIRST_ROTATING f', TURRET_FIRST_ROTATING)
        if STATE_DEBUG : print('3 🤩🤩re action - TURRET_HIT 0', TURRET_HIT)
    else:
        if DEBUG: print("💥 Hit!!!!!")
        TURRET_HIT = 1
        action_command = turret.get_reverse_action_command(
            player_data.get('turret_x', 0),
            player_data.get('turret_y', 0),
        )
        
        if STATE_DEBUG : print('4 🤩🤩reverse - TURRET_FIRST_ROTATING f', TURRET_FIRST_ROTATING)
        if STATE_DEBUG : print('4 🤩🤩reverse - TURRET_HIT 1', TURRET_HIT)

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
        destination = {'x': x, 'y': y, 'z': z}

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
    global TURRET_FIRST_ROTATING, TURRET_HIT
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