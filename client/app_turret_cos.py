# app.py
# 상태 정의
# 45m 이내 목표물 도달 시 상태를 ROTATING으로 전환하고 자동으로 360도 회전시작 
# 회전 중 적을 탐지하면 FIRING 상태로 전환, 회전 중단 및 포격 수행
# 포격 후 명중 여부에 따라 다음 타겟으로 진행하거나 IDLE 상태로 전환
from flask import Flask, request, jsonify, render_template
import logging
from flask_socketio import SocketIO
from PIL import Image, ImageDraw, ImageFont
import torch
from ultralytics import YOLO
import time
import json
import modules.turret as turret
import modules.is_near_enemy_pract as is_near_enemy
import math

app = Flask(__name__)
socketio = SocketIO(app)

DEBUG = True

# 전역 상태 변수들
move_command = []
action_command = []
player_data = {'pos': {'x': 60, 'y': 10, 'z': 57}}
destination = None
impact_info = {}
obstacles = []
latest_nearest_enemy = None
enemy_queue = []
first_action_state = True
hit_state = -1
MATCH_THRESHOLD = 3.0

# FSM 상태
STATE = 'IDLE'  # 가능 상태: IDLE, ROTATING, FIRING, PAUSE

# 스테레오 카메라 초기 위치 설정
stereo_config = {
    "StereoL_X": 9.9,
    "StereoL_Y": 10.0,
    "StereoL_Z": 10.0,
    "StereoL_Roll": 0.0,
    "StereoL_Pitch": 0.0,
    "StereoL_Yaw": 0.0,
    "StereoR_X": 10.1,
    "StereoR_Y": 10.0,
    "StereoR_Z": 10.0,
    "StereoR_Roll": 0.0,
    "StereoR_Pitch": 0.0,
    "StereoR_Yaw": 0.0
}

# YOLO 모델 로드
try:
    model = YOLO('best.pt')
except Exception as e:
    raise RuntimeError(f"YOLO model loading failed: {str(e)}")

# 필터링
EXCLUDE_PATHS = ("/info", "/start", "/update_position", "/get_move", "/get_action")
class PathFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return not any(path in msg for path in EXCLUDE_PATHS)
log = logging.getLogger("werkzeug")
log.addFilter(PathFilter())

@app.route('/streo_config', methods=['GET'])
def get_stereo_config():
    return jsonify(stereo_config)

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/detect', methods=['POST'])
def detect():
    global STATE, player_data, obstacles, latest_nearest_enemy, action_command
    global destination, first_action_state, enemy_queue, is_rotating
    
    image = request.files.get('image')
    if not image:
        return jsonify({"error": "No image received"}), 400
    
    print("🌍 detect called")
    image_path = 'temp_image.jpg'
    image.save(image_path)

    results = model(image_path, imgsz=640)
    detections = results[0].boxes.data.cpu().numpy()

    target_classes = {
        0: 'car002', 1: 'car003', 2: 'car005', 3: 'human001',
        4: 'rock001', 5: 'rock2', 6: 'tank', 7: 'wall001', 8: 'wall002'
    }

    filtered_results = []
    for box in detections:
        class_id = int(box[5])
        if class_id in target_classes:
            filtered_results.append({
                'className': target_classes[class_id],
                'bbox': [float(coord) for coord in box[:4]],
                'confidence': float(box[4]),
                'color': '#00FF00',
                'filled': False,
                'updateBoxWhileMoving': False
            })
    player_pos = (
        player_data.get('pos', {}).get('x', 60),
        player_data.get('pos', {}).get('z', 57)
    )

    best_target = None
    for detection in filtered_results:
        target = is_near_enemy.match_detection_to_obstacle(
            detection=detection,
            player_pos=player_data['pos'],
            obstacles=obstacles,
            image_width=1920,
            image_height=1080
        )
        if target:
            best_target = target
            break  # 첫 번째 유효 타겟만 선택 (여러 개 처리하고 싶다면 리스트로 바꿔도 됨)

    if best_target:
        print("🎯 Firing at:", best_target)
        return jsonify({
            "target": best_target,
            "status": "ready"
        })
    else:
        print("❌ No matching target found.")
        return jsonify({
            "target": None,
            "status": "not_found"
        })
    return jsonify(filtered_results)

@app.route('/update_bullet', methods=['POST'])
def update_bullet():
    global STATE, destination, impact_info, player_data, action_command, latest_nearest_enemy, enemy_queue
    data = request.get_json()
    if not data:
        return jsonify({"status": "ERROR", "message": "Invalid request data"}), 400
    impact = {
        'x': data.get('x'),
        'y': data.get('y'),
        'z': data.get('z'),
        'target': data.get('hit')
    }
    is_hit = turret.is_hit(latest_nearest_enemy, impact)

    if is_hit:
        print("🎯 Target HIT confirmed.")
        if enemy_queue:
            latest_nearest_enemy = enemy_queue.pop(0)
            action_command = turret.get_action_command(
                player_data['pos'], latest_nearest_enemy,
                turret_x_angle=player_data['turret_x'],
                turret_y_angle=player_data['turret_y'],
                player_y_angle=player_data['body_y']
            )
            STATE = "FIRING"
        else:
            action_command = turret.get_reverse_action_command(
                player_data['turret_x'], player_data['turret_y'], player_data['body_y']
            )
            STATE = "IDLE"
    else:
        print("💫 Missed. Re-firing...")
        action_command = turret.get_action_command(
            player_data['pos'], latest_nearest_enemy,
            turret_x_angle=player_data['turret_x'],
            turret_y_angle=player_data['turret_y'],
            player_y_angle=player_data['body_y']
        )
        STATE = "FIRING"

    return jsonify({"status": "OK", "result": "Processed"})

@app.route('/set_destination', methods=['POST'])
def set_destination():
    global destination, action_command
    if DEBUG: print('🚨 set_destination >>>')
    data = request.get_json()
    action_command = []
    if not data or "destination" not in data:
        if DEBUG: print("🚫 Missing destination data")
        return jsonify({"status": "ERROR", "message": "Missing destination data"}), 400

    try:
        x, y, z = map(float, data["destination"].split(","))
        destination = {'x': x, 'y': y, 'z': z}
        if DEBUG: print(f"🎯 Destination set to: {destination}")
        action_command = turret.get_action_command(
            player_data.get('pos', {'x': 60, 'y': 10, 'z': 57}),
            destination,
            turret_x_angle=player_data.get('turret_x', 0),
            turret_y_angle=player_data.get('turret_y', 0),
            player_y_angle=player_data.get('body_y', 0)
        )
        if DEBUG: print('action_command:', action_command)
        return jsonify({"status": "OK", "destination": destination})
    except Exception as e:
        if DEBUG: print(f"🚫 Invalid destination format: {str(e)}")
        return jsonify({"status": "ERROR", "message": f"Invalid format: {str(e)}"}), 400

@app.route('/get_action', methods=['GET'])
def get_action():
    global STATE, action_command
    if STATE == "PAUSE":
        return jsonify({"turret": "", "weight": 0.0})
    if action_command:
        cmd = action_command.pop(0)
        if cmd['turret'] == "FIRE":
            STATE = "PAUSE"
            if DEBUG : print("💥 FIRE issued. STATE -> PAUSE")
        return jsonify(cmd)
    return jsonify({"turret": "", "weight": 0.0})

@app.route('/info', methods=['POST'])
def info():
    global player_data
    data = request.get_json(force=True)

    player_data = {
        'pos': data.get('playerPos', player_data.get('pos')),
        'turret_x': data.get('playerTurretX', player_data.get('turret_x', 0)),
        'turret_y': data.get('playerTurretY', player_data.get('turret_y', 0)),
        'body_x': data.get('playerBodyX', player_data.get('body_x', 0)),
        'body_y': data.get('playerBodyY', player_data.get('body_y', 0)),
        'body_z': data.get('playerBodyZ', player_data.get('body_z', 0)),
    }
    return jsonify({"status": "success"})

@app.route('/update_obstacle', methods=['POST'])
def update_obstacle():
    global obstacles
    data = request.get_json()
    obstacle_list = data.get('obstacles', [])
    print(type(data))
    for obs in obstacle_list:
        center_x = (obs['x_min'] + obs['x_max'])/2
        center_z = (obs['z_min'] + obs['z_max'])/2
        print(f"🪨obstacles Center: x = {center_x:.2f}, z= {center_z:.2f}")
    obstacles = obstacle_list
    return jsonify({'status': 'success'})

@app.route('/collision', methods=['POST']) 
def collision():
    data = request.get_json()
    if not data:
        return jsonify({'status': 'error', 'message': 'No collision data received'}), 400

    object_name = data.get('objectName')
    position = data.get('position', {})
    x = position.get('x')
    y = position.get('y')
    z = position.get('z')

    print(f"💥 Collision Detected - Object: {object_name}, Position: ({x}, {y}, {z})")

    return jsonify({'status': 'success', 'message': 'Collision data received'})

@app.route('/init', methods=['GET'])
def init():
    if DEBUG: print('🚨 init >>>')

    config = {
        "startMode": "start",
        "blStartX": 60,
        "blStartY": 10,
        "blStartZ": 57,
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
    if DEBUG: print(f"🛠️ Initialization config sent via /init: {config}")
    return jsonify(config)

@app.route('/start', methods=['GET'])
def start():
    if DEBUG: print("🚀 /start command received")
    return jsonify({"control": ""})

@app.route('/start_rotation', methods=['POST'])
def start_rotation():
    global is_rotating, action_command
    print("🛞rotation start!")
    if STATE != 'ROTATING':
        STATE = 'ROTATING'
        is_rotating = True
        action_command = []
        for _ in range(36):
            action_command.append({"turret": 'Q', "weight": 0.5})
        action_command.append({"turret": 'Q', "weight": 0.0})
    return jsonify({"status": "OK", "message": "Rotation started"})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
