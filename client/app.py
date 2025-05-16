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
import modules.rotation_controller as rotation_controller
import math
import os


app = Flask(__name__)

DEBUG = True
STATE_DEBUG = False

# YOLO 모델 로드
try:
    model = YOLO('best_add.pt')

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

@app.route('/dashboard')
def dashboard():
    if DEBUG: print('🚨 dashboard >>>')
    return render_template('dashboard.html')

STATE = 'PAUSE'
TURRET_FIRST_ROTATING = True
TURRET_HIT = -1

@app.route('/detect', methods=['POST'])
def detect():
    global player_data, latest_nearest_enemy, action_command, destination, obstacles_from_map
    global TURRET_FIRST_ROTATING, TURRET_HIT
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
        4: 'Rock001', 5: 'Tank001', 6: 'Wall001'
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

    if STATE_DEBUG : print('1 🤩🤩TURRET_FIRST_ROTATING', TURRET_FIRST_ROTATING)
    if STATE_DEBUG : print('1 🤩🤩TURRET_HIT', TURRET_HIT)

    nearest_enemy = {'state': False}
    enemy_list = []
    dead_enemy_list = []
    if STATE == 'PAUSE':
        # enemy_list = get_enemy_pos.get_enemy_list(filtered_results, player_data, obstacles_from_map)
        nearest_enemy = get_enemy_pos.find_nearest_enemy(filtered_results, player_data, obstacles_from_map)
    print('📀 nearest_enemy', nearest_enemy)
    if nearest_enemy['state'] and TURRET_FIRST_ROTATING:
        try:
            # if DEBUG: print(f"👉 Generating action command: player_pos={player_data.get('pos')}, dest={destination}")
            latest_nearest_enemy = nearest_enemy
            action_command = turret.get_action_command(
                player_data['pos'],
                nearest_enemy,
                turret_x_angle=player_data['turret_x'],
                turret_y_angle=player_data['turret_y'],
                player_y_angle=player_data['body_y']
            )
        
            print('📀 action_command', action_command)
        except ValueError as e:
            print(f"🚫 Error generating action command: {str(e)}")
            action_command = []
        
        if STATE_DEBUG : print('2 🤩🤩action - TURRET_FIRST_ROTATING f', TURRET_FIRST_ROTATING)
        if STATE_DEBUG : print('2 🤩🤩action - TURRET_HIT -1', TURRET_HIT)

    return jsonify(filtered_results)

@app.route('/info', methods=['POST'])
def info():
    if DEBUG: print('🚨 info >>>')
    global player_data
    data = request.get_json(force=True)
    if not data:
        if DEBUG: print("🚫 No JSON received")
        return jsonify({"error": "No JSON received", "control": ""}), 400
    
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
    if DEBUG: print(f"📍 Player data updated: {player_data}")
    return jsonify({"status": "success", "control": ""})

# @app.route('/update_position', methods=['POST'])
# def update_position():
#     global player_data
#     if DEBUG: print('🚨 update_position >>>')
#     data = request.get_json()
#     if not data or "position" not in data:
#         if DEBUG: print("🚫 Missing position data")
#         return jsonify({"status": "ERROR", "message": "Missing position data"}), 400

#     try:
#         x, y, z = map(float, data["position"].split(","))
#         player_data['pos'] = {'x': x, 'y': y, 'z': z}
#         player_data.setdefault('turret_x', 0)
#         player_data.setdefault('turret_y', 0)
#         player_data.setdefault('body_x', 0)
#         player_data.setdefault('body_y', 0)
#         player_data.setdefault('body_z', 0)
#         if DEBUG: print(f"📍 Position updated: {player_data['pos']}")
#         return jsonify({"status": "OK", "current_position": player_data['pos']})
#     except Exception as e:
#         if DEBUG: print(f"🚫 Invalid position format: {str(e)}")
#         return jsonify({"status": "ERROR", "message": str(e)}), 400

# @app.route('/get_move', methods=['GET'])
# def get_move():
#     if DEBUG: print('🚨 get_move >>>')
#     global move_command
#     if move_command:
#         command = move_command.pop(0)
#         if DEBUG: print(f"🚗 Move Command: {command}")
#         return jsonify(command)
#     else:
#         return jsonify({"move": "STOP", "weight": 1.0})
    
@app.route('/get_action', methods=['POST'])
def get_action():
    global TURRET_FIRST_ROTATING, TURRET_HIT, MOVING
    global action_command, latest_nearest_enemy, is_rotating
    data = request.get_json(force=True)

    position = data.get("position", {})
    turret = data.get("turret", {})

    pos_x = position.get("x", 0)
    pos_y = position.get("y", 0)
    pos_z = position.get("z", 0)

    turret_x = turret.get("x", 0)
    turret_y = turret.get("y", 0)

    print(f"📨 Position received: x={pos_x}, y={pos_y}, z={pos_z}")
    print(f"🎯 Turret received: x={turret_x}, y={turret_y}")
    
    if action_command:
        TURRET_FIRST_ROTATING = False
        command = action_command.pop(0)
        start_rotation()
        if DEBUG: print(f"🔫 Action Command: {command}")
        
        if TURRET_HIT == 1 and command['turretQE']['command'] == 'STOP':
            # reverse 끝나는 지점
            TURRET_FIRST_ROTATING = True
            TURRET_HIT = -1
            MOVING = 'MOVING'
            # print("impact_control False", action_command)
            if STATE_DEBUG : print('5 🤩🤩reverse end - TURRET_FIRST_ROTATING t', TURRET_FIRST_ROTATING)
            if STATE_DEBUG : print('5 🤩🤩reverse end - TURRET_HIT -1', TURRET_HIT)

    else:
        # rotation_controller.degree_check_and_stop(player_data, action_command)
        # print("최종 정지!")
        command = {
            "moveWS": {"command": "STOP", "weight": 1.0},
            "moveAD": {"command": "", "weight": 0.0},
            "turretQE": {"command": "", "weight": 0.0},
            "turretRF": {"command": "", "weight": 0.0},
            "fire": False
        }

    return jsonify(command)

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
        'hit' : None,
        'timestamp': time.strftime('%H:%M:%S'),
        'tx' : latest_nearest_enemy.get('x'),
        'ty' : latest_nearest_enemy.get('y'),
        'tz' : latest_nearest_enemy.get('z'),
    }

    is_hit = turret.is_hit(latest_nearest_enemy, impact_info)
    impact_info['hit'] = is_hit

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
            player_data.get('body_x', 0),
            player_data.get('body_y', 0),
        )
        
        if STATE_DEBUG : print('4 🤩🤩reverse - TURRET_FIRST_ROTATING f', TURRET_FIRST_ROTATING)
        if STATE_DEBUG : print('4 🤩🤩reverse - TURRET_HIT 1', TURRET_HIT)

    socketio.emit('bullet_impact', impact_info)
    return jsonify({"status": "OK", "message": "Bullet impact data received"})

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
    map_path = 'NewMap.map'
    obstacles_from_map = get_obstacles.load_obstacles_from_map(map_path)
    print('obstacles_from_map', obstacles_from_map)
    return jsonify({"control": ""})

def get_player_data():
    return player_data

@app.route('/start_rotation', methods=['POST'])
def start_rotation():
    print("🫡🛞/start_rotation")
    global action_command, player_data
    print("BEFORE",player_data)
    import threading
    threading.Thread(target=rotation_controller.start_rotation_10degree, args=(get_player_data, action_command)).start()
    print("AFTER", player_data)
    return jsonify({"status": "started"})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)