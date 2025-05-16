from flask import Flask, request, jsonify, render_template
import logging
from flask_socketio import SocketIO
from PIL import Image, ImageDraw, ImageFont
import torch
from ultralytics import YOLO
import time
import json
import modules.turret_list as turret
import modules.get_enemy_pos_list as get_enemy_pos
import modules.get_obstacles as get_obstacles
import math
import os
from collections import deque

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
dead_list = [] # 명중된 탱크 리스트
enemy_queue = deque()  # 적 위치를 저장할 큐
MAX_FIRE_ATTEMPS = 3  # 최대 발사 시도 횟수

@app.route('/dashboard')
def dashboard():
    if DEBUG: print('🚨 dashboard >>>')
    return render_template('dashboard.html')

MOVING = 'PAUSE'
TURRET_FIRST_ROTATING = True
TURRET_HIT = -1

@app.route('/detect', methods=['POST'])
@app.route('/detect', methods=['POST'])
def detect():
    global player_data, latest_nearest_enemy, action_command, destination, obstacles_from_map
    global TURRET_FIRST_ROTATING, TURRET_HIT, enemy_queue

    print('🌍 detect >>>')

    image = request.files.get('image')
    if not image:
        return jsonify({"error": "No image received"}), 400

    image_path = 'temp_image.jpg'
    image.save(image_path)

    results = model(image_path, imgsz=640)
    detections = results[0].boxes.data.cpu().numpy()

    target_classes = {
        0: 'Car002', 1: 'Car003', 2: 'Car005', 3: 'Human001',
        4: 'Rock001', 5: 'Rock2', 6: 'Tank001', 7: 'Wall001', 8: 'Wall002'
    }

    filtered_results = []
    for index, box in enumerate(detections):
        class_id = int(box[5])
        if class_id not in target_classes:
            continue
        filtered_results.append({
            'id': index,
            'className': target_classes[class_id],
            'bbox': [float(x) for x in box[:4]],
            'confidence': float(box[4]),
            'color': '#0000FF',
            'filled': False
        })

    print("Player position:", player_data['pos'])

    nearest_enemy = {'state': False}
    if MOVING == 'PAUSE':
        nearest_enemy = get_enemy_pos.find_nearest_enemy(filtered_results, player_data, obstacles_from_map)
    print('📀 nearest_enemy', nearest_enemy)

    # 🎯 새로운 적 리스트 생성
    all_tanks = get_enemy_pos.get_enemy_list(filtered_results, player_data, obstacles_from_map)
    print("🪡", all_tanks)

    if isinstance(all_tanks, list) and all_tanks:
        enemy_queue.clear()
        print("🧹 Cleared enemy_queue")
        existing_keys = set()
        for enemy in all_tanks:
            if enemy['className'] != 'Tank001':
                continue

            # 💀 명중한 적 제거
            if any(abs(dead['x'] - enemy['x']) < 1.0 and abs(dead['z'] - enemy['z']) < 1.0 for dead in dead_list):
                print(f"💀 Skipping dead enemy: x={enemy['x']:.1f}, z={enemy['z']:.1f}")
                continue

            # 🔁 위치 기준 중복 제거
            key = f"{round(enemy['x'], 2)}_{round(enemy['z'], 2)}"
            if key in existing_keys:
                continue
            existing_keys.add(key)

            enemy['id'] = f"Tank001_{key}"
            enemy['fire_count'] = 0
            enemy_queue.append(enemy)
            print(f"🧹 Added enemy to queue: {enemy['id']} at {enemy['x']}, {enemy['z']}")
    else:
        print("⚠️ all_tanks is not a valid list — skipping queue update")

    if nearest_enemy['state'] and TURRET_FIRST_ROTATING:
        try:
            latest_nearest_enemy = nearest_enemy
            action_command = turret.get_action_command(
                player_data['pos'],
                nearest_enemy,
                turret_x_angle=player_data.get('turret_x', 0),
                turret_y_angle=player_data.get('turret_y', 0),
                player_y_angle=player_data.get('body_y', 0)
            )
            print('📀 action_command', action_command)
        except ValueError as e:
            print(f"🚫 Error generating action command: {str(e)}")
            action_command = []

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
    # if DEBUG: print(f"📍 Player data updated: {player_data}")
    return jsonify({"status": "success", "control": ""})

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
    global TURRET_FIRST_ROTATING, TURRET_HIT, MOVING
    global action_command, latest_nearest_enemy, enemy_queue

    if DEBUG: print('🚨 get_action >>>', action_command)

    if action_command:
        TURRET_FIRST_ROTATING = False
        command = action_command.pop(0)
        if DEBUG: print(f"🔫 Action Command: {command}")
        return jsonify(command)

    if not enemy_queue or TURRET_FIRST_ROTATING:
        return jsonify({"turret": "", "weight": 0.0})

    target = enemy_queue[0]
    print(f"🎯 Current target: id={target['id']}, fire_count={target.get('fire_count', 0)}")

    if 'fire_count' not in target:
        target['fire_count'] = 0

    if target['fire_count'] >= MAX_FIRE_ATTEMPS:
        print(f"🔥 MAX attempt reached — removing: {target['id']}")
        enemy_queue.popleft()
        return jsonify({"turret": "", "weight": 0.0})

    try:
        action_command = turret.get_action_command(
            player_data['pos'],
            target,
            turret_x_angle=player_data.get('turret_x', 0),
            turret_y_angle=player_data.get('turret_y', 0),
            player_y_angle=player_data.get('body_y', 0)
        )
        target['fire_count'] += 1
        print(f"💥 Firing at {target['id']}, fire_count={target['fire_count']}")
    except ValueError as e:
        print(f"🚫 Action gen failed for {target['id']}: {e}")
        enemy_queue.popleft()
        return jsonify({"turret": "", "weight": 0.0})

    if action_command:
        return jsonify(action_command.pop(0))
    else:
        print("⚠️ Empty action_command generated.")
        return jsonify({"turret": "", "weight": 0.0})
    

    # if TURRET_HIT == 0:  
    #     if TURRET_HIT == 1 and command['turret'] != 'FIRE' and command['weight'] == 0.0:
    #         # reverse 끝나는 지점
    #         TURRET_FIRST_ROTATING = True
    #         TURRET_HIT = -1
    #         MOVING = 'MOVING'
    #         # print("impact_control False", action_command)
    #         if STATE_DEBUG : print('5 🤩🤩reverse end - TURRET_FIRST_ROTATING t', TURRET_FIRST_ROTATING)
    #         if STATE_DEBUG : print('5 🤩🤩reverse end - TURRET_HIT -1', TURRET_HIT)

    #     return jsonify(command)
    # else:
    #     return jsonify({"turret": "", "weight": 0.0})

def print_enemy_queue():
    print(f"📋 [QUEUE STATUS] {len(enemy_queue)} enemies in queue:")
    for i, enemy in enumerate(enemy_queue):
        print(f"  {i+1}. id={enemy['id']}, fire_count={enemy['fire_count']}, x={enemy['x']:.1f}, z={enemy['z']:.1f}")


@app.route('/update_bullet', methods=['POST'])
def update_bullet():
    global destination, impact_info, player_data, action_command, latest_nearest_enemy, TURRET_HIT
    if DEBUG: print('🚨 update_bullet >>>')
    data = request.get_json()
    action_command = []

    if not data:
        return jsonify({"status": "ERROR", "message": "Invalid request data"}), 400

    print(f"💥 Bullet Impact at X={data.get('x')}, Y={data.get('y')}, Z={data.get('z')}, Target={data.get('hit')}")
    impact_info = {
        'x': data.get('x'),
        'y': data.get('y'),
        'z': data.get('z'),
        'target': data.get('hit'),
        'hit' : None,
        'timestamp': time.strftime('%H:%M:%S'),
        'tx' : latest_nearest_enemy.get('x') if latest_nearest_enemy else None,
        'ty' : latest_nearest_enemy.get('y') if latest_nearest_enemy else None,
        'tz' : latest_nearest_enemy.get('z') if latest_nearest_enemy else None,
    }

    if not latest_nearest_enemy:
        print("❌ latest_nearest_enemy is None — cannot check hit.")
        return jsonify({"status": "OK", "message": "Skipped due to missing enemy data"})

    is_hit = turret.is_hit(latest_nearest_enemy, impact_info)
    impact_info['hit'] = is_hit

    if DEBUG: print('💥', is_hit)

    if not is_hit:
        TURRET_HIT = 0
        # 🧤 현재 회전 정보가 없으면 return
        if any(player_data.get(k) is None for k in ['turret_x', 'turret_y', 'body_y']):
            print("⚠️ Missing rotation info — skipping re-aim")
            return jsonify({"status": "OK", "message": "Skipped due to missing turret angles"})

        time.sleep(3)  # 이건 향후 async로 개선
        try:
            action_command = turret.get_action_command(
                player_data['pos'],
                latest_nearest_enemy,
                impact_info,
                turret_x_angle=player_data.get('turret_x'),
                turret_y_angle=player_data.get('turret_y'),
                player_y_angle=player_data.get('body_y')
            )
        except ValueError as e:
            print(f"🚫 Re-aim failed: {e}")
            action_command = []

    else:
        TURRET_HIT = 1
        print("💥 Hit!!!!!")
        dead_list.append({'x': impact_info['x'], 'z': impact_info['z']})
        print(f"💀 Added to dead_list: {impact_info['x']:.1f}, {impact_info['z']:.1f}")
        dead_id = latest_nearest_enemy.get('id')
        enemy_queue = deque([e for e in enemy_queue if e['id'] != dead_id])
        print(f"🧹 Removed hit target from queue: id={dead_id}")

        action_command = turret.get_reverse_action_command(
            player_data.get('turret_x', 0),
            player_data.get('turret_y', 0),
            player_data.get('body_x', 0),
            player_data.get('body_y', 0),
        )

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
    map_path = './NewMap.map'
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