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
import math
import pdb
import threading
import requests

app = Flask(__name__)

DEBUG = True
STATE_DEBUG = True
 

# YOLO 모델 로드

model = YOLO('./best.pt')

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
obstacles = []  # /set_obstacles 데이터 저장
obstacles_center = []
latest_nearest_enemy = None
MATCH_THRESHOLD = 3.0

#정적인 적 - 가까운 적을 타격한 것을 표시하고 더이상 쏘지 않게 한다.
dead_list =[]

#3 FOV 및 카메라 설정
FOV_HORIZONTAL = 50
FOV_VERTICAL = 28
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
MAP_WIDTH = 300
MAP_HEIGHT = 300
score = 0

@app.route('/dashboard')
def dashboard():
    if DEBUG: print('🚨 dashboard >>>')
    return render_template('dashboard.html')

first_action_state = True
hit_state = -1
@app.route('/detect', methods=['POST'])
def detect():
    global player_data, obstacles, latest_nearest_enemy, action_command, destination, first_action_state, hit_state
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
        0: 'car002', 1: 'car003', 2: 'car005', 3: 'human001',
        4: 'rock001', 5: 'rock2', 6: 'tank', 7: 'wall001', 8: 'wall002'
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

    if STATE_DEBUG : print('1 🤩🤩first_action_state', first_action_state)
    if STATE_DEBUG : print('1 🤩🤩hit_state', hit_state)

    nearest_enemy = get_enemy_pos.find_nearest_enemy(filtered_results, player_data, obstacles)
    print("🔍 nearest_enemy result:", nearest_enemy)
    if nearest_enemy['state'] and first_action_state:
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
            # first_action_state = False
        except ValueError as e:
            print(f"🚫 Error generating action command: {str(e)}")
            action_command = []
        
        if STATE_DEBUG : print('2 🤩🤩action - first_action_state f', first_action_state)
        if STATE_DEBUG : print('2 🤩🤩action - hit_state -1', hit_state)

    # 💣 dead_list에 있는 적이면 무시
    if nearest_enemy['state']:
        ex = nearest_enemy['x']
        ez = nearest_enemy['z']
        if get_enemy_pos.is_already_dead(ex, ez, dead_list):
            print("🧟‍♂️ 이미 사망한 타겟. 포격 제외.")
            return jsonify({"status": "already_dead", "target": None})
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

        if destination:
            dx = x - destination['x']
            dz = z - destination['z']
            distance =math.sqrt(dx**2 + dz**2)

            if distance < 45:
                print("🎯 목적지 도착! 자동 회전 시작.")
                start_rotation()
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
    global action_command, latest_nearest_enemy, first_action_state, hit_state
    if DEBUG: print('🚨 get_action >>>', action_command)
    if action_command:
        first_action_state = False
        command = action_command.pop(0)
        if DEBUG: print(f"🔫 Action Command: {command}")
        
        if hit_state == 1 and command['turret'] != 'FIRE' and command['weight'] == 0.1:
            # reverse 끝나는 지점
            first_action_state = True
            hit_state = -1
            # print("impact_control False", action_command)
            if STATE_DEBUG : print('5 🤩🤩reverse end - first_action_state t', first_action_state)
            if STATE_DEBUG : print('5 🤩🤩reverse end - hit_state -1', hit_state)

        return jsonify(command)
    else:
        return jsonify({"turret": "", "weight": 0.0})

@app.route('/update_bullet', methods=['POST'])
def update_bullet():
    global destination, impact_info, player_data, action_command, latest_nearest_enemy, hit_state, score
    if DEBUG: print('🚨 update_bullet >>>')
    data = request.get_json()
    action_command = []
    if not data:
        if DEBUG: print("🚫 Invalid bullet data")
        return jsonify({"status": "ERROR", "message": "Invalid request data"}), 400

    if not latest_nearest_enemy:
        print("⚠️ No valid enemy to compare bullet impact. Skipping is_hit() check.")
        return jsonify({"status": "skipped", "message": "No target set"})

    print(f"💥 Bullet Impact at X={data.get('x')}, Y={data.get('y')}, Z={data.get('z')}, Target={data.get('hit')}")
    impact_info = {
        'x': data.get('x'),
        'y': data.get('y'),
        'z': data.get('z'),
        'target': data.get('hit'),
        'timestamp': time.strftime('%H:%M:%S')
    }

    is_hit = turret.is_hit(latest_nearest_enemy, impact_info)
    hit_target = impact_info.get("target", "").lower()
    print("💕💕💕hit_target", hit_target)
    excepted_target = latest_nearest_enemy.get("className","").lower()
    if DEBUG: print('💥', is_hit)
     # 🎯 리워드/패널티 로직
    if is_hit:
        if "tank" in hit_target:
            score += 10  # 적 맞춤 → 보상
            print("✅ 적 명중! +10점")

        else:
            score -= 10  # 아군 명중 → 패널티
            print("❌ 아군 명중! -10점")
        hit_state = 1
        dead_list.append({"x": impact_info['x'], "z": impact_info['z']})
        action_command = turret.get_reverse_action_command(
            player_data.get('turret_x', 0),
            player_data.get('turret_y', 0),
            player_data.get('body_x', 0),
            player_data.get('body_y', 0),
        )
    else:
        if "tank" in hit_target:
            score -= 5  # 적 놓침 → 패널티
            print("❌ 적을 놓침! -5점")
        else:
            score += 5  # 아군 안 맞춤 → 보상
            print("✅ 아군 안 맞춤! +5점")
        hit_state = 0
        time.sleep(5)
        try:
            action_command = turret.get_action_command(
                player_data.get('pos', {'x': 60, 'y': 10, 'z': 57}),
                latest_nearest_enemy,
                turret_x_angle=player_data.get('turret_x', 0),
                turret_y_angle=player_data.get('turret_y', 0),
                player_y_angle=player_data.get('body_y', 0)
            )
        except ValueError as e:
            action_command = []

    print(f"📊 현재 점수: {score}")
    socketio.emit('bullet_impact', impact_info)
    # 명중 못했을 때 
    if not is_hit:
        hit_state = 0
        time.sleep(5)
        try:
            action_command = turret.get_action_command(
                player_data.get('pos', {'x': 60, 'y': 10, 'z': 57}),
                latest_nearest_enemy,
                turret_x_angle=player_data.get('turret_x', 0),
                turret_y_angle=player_data.get('turret_y', 0),
                player_y_angle=player_data.get('body_y', 0)
            )
            if DEBUG: print('💥 is_hit >> action_command:', action_command)
        except ValueError as e:
            if DEBUG: print(f"🚫 Error generating action command: {str(e)}")
            action_command = []
        
        if STATE_DEBUG : print('3 🤩🤩re action - first_action_state f', first_action_state)
        if STATE_DEBUG : print('3 🤩🤩re action - hit_state 0', hit_state)
    else:
        if DEBUG: print("💥 Hit!!!!!")
        hit_state = 1
        if is_hit:
            print("🎯 Target HIT confirmed.")
            # 💀 dead_list에 등록
            dead_list.append({
                "x" : impact_info['x'],
                "z": impact_info['z']
            })
            print(dead_list)
        action_command = turret.get_reverse_action_command(
            player_data.get('turret_x', 0),
            player_data.get('turret_y', 0),
            player_data.get('body_x', 0),
            player_data.get('body_y', 0),
        )
        
        if STATE_DEBUG : print('4 🤩🤩reverse - first_action_state f', first_action_state)
        if STATE_DEBUG : print('4 🤩🤩reverse - hit_state 1', hit_state)

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
    global obstacles, obstacles_center
    if DEBUG: print('🚨 update_obstacle >>>')
    data = request.get_json()
    if not data or 'obstacles' not in data:
        if DEBUG: print("🚫 No obstacle data received")
        logging.warning("No obstacle data received")
        return jsonify({'status': 'error', 'message': 'No data received'}), 400

    obstacles = data['obstacles']
    print(f"🪨 Obstacle data updated: {obstacles}")
    # logging.debug(f"Obstacle data updated: {json.dumps(obstacles, indent=2)}")
    # if DEBUG: print(f"🪨 Obstacle data: {json.dumps(obstacles, indent=2)}")
    return jsonify({'status': 'success', 'message': 'Obstacle data received', 'obstacles_count': len(obstacles)})

def load_map_to_obstacles(map_path='modules/test_turret_test.map'):
    import json
    import os

    if not os.path.exists(map_path):
        print(f"❌ Map file not found: {map_path}")
        return []

    with open(map_path, 'r') as f:
        map_data = json.load(f)

    prefab_size = {
        'Car002': (3.0, 3.0),
        'Car003': (3.0, 3.0),
        'Tank': (4.0, 4.0),
        'Rock001': (2.0, 2.0),
        'Wall001': (5.0, 1.0),
        # 필요시 추가
    }

    converted = []
    for obj in map_data.get('obstacles', []):
        name = obj['prefabName']
        pos = obj['position']
        width, depth = prefab_size.get(name, (2.0, 2.0))
        converted.append({
            "x_min": pos['x'] - width / 2,
            "x_max": pos['x'] + width / 2,
            "z_min": pos['z'] - depth / 2,
            "z_max": pos['z'] + depth / 2,
            "y_center": pos['y'], # y좌표 추가
            "className": name.lower(),
            "center": (pos['x'], pos['y'], pos['z'],)
        })

    return converted

@app.route('/init', methods=['GET'])
def init():
    global first_action_state, hit_state
    if DEBUG: print('🚨 init >>>')

    config = {
        "startMode": "start",
        "blStartX": 60,
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

    first_action_state = True
    hit_state = -1

    if DEBUG: print(f"🛠️ Initialization config sent via /init: {config}")
    return jsonify(config)

@app.route('/start', methods=['GET'])
def start():
    global obstacles
    if DEBUG: print("🚀 /start command received")
    map_path = 'modules/test_turret.map'
    obstacles = load_map_to_obstacles(map_path)
    # print(obstacles)
    print(f"🗺️ Map loaded: {len(obstacles)} obstacles from {map_path}")

    return jsonify({"control": "", "message": f"{len(obstacles)} obstacles loaded from map"})

def wait_for_impact_confirm(timeout=3.0):
    """/update_bullet로 명중 여부가 반영될 때까지 기다림"""
    global hit_state
    start_time = time.time()
    print("⏳ 포격 후 명중 여부 확인 중...")

    while time.time() - start_time < timeout:
        if hit_state in [0, 1]:  # 0=miss, 1=hit
            print(f"✅ 명중 여부 확인 완료: hit_state={hit_state}")
            return
        time.sleep(0.1)  # 100ms 단위로 확인

    print("⚠️ 제한 시간 내 명중 여부 확인 실패")

def auto_start_rotation():
    time.sleep(1)
    requests.post("http://localhost:5000/start_rotation")

@app.route('/start_rotation', methods=['POST'])
def start_rotation():
    global action_command,  player_data, obstacles, dead_list, latest_nearest_enemy
    print('🚨 start_rotation >>>')
    if DEBUG: print('🚨 start_rotation >>>')

    for _ in range(36):  # 360도 회전 (10도씩)
        # 1. 회전 명령 큐에 추가 (Q: 좌회전)
        action_command.append({"turret": "Q", "weight": 0.1})
        action_command.append({"turret": "Q", "weight": 0.0})  # 회전 멈춤

        # 2. YOLO 탐지 → 큐 생성
        image_path = 'temp_image.jpg'
        try:
            results = model(image_path, imgsz=640)
            detections = results[0].boxes.data.cpu().numpy()
        except Exception as e:
            print(f"❌ YOLO 실패: {e}")
            continue  # 예외 발생 시 다음 회전으로

        # 3. 탐지 결과 처리
        filtered_results = []
        target_classes = {0: 'car002', 1: 'tank'}
        for box in detections:
            class_id = int(box[5])
            if class_id not in target_classes:
                continue
            bbox = [float(coord) for coord in box[:4]]
            filtered_results.append({
                'className': target_classes[class_id],
                'bbox': bbox,
                'confidence': float(box[4])
            })

        # 3. 가장 가까운 적 탐색
        enemy_queue = get_enemy_pos.find_all_valid_enemies(filtered_results, player_data, obstacles)
        print(f"🎯 유효 타겟 수: {len(enemy_queue)}")

    # 4. 하나씩 타겟을 꺼내서 포격
        for enemy in enemy_queue:
            ex, ez = enemy['x'], enemy['z']
            if get_enemy_pos.is_already_dead(ex, ez, dead_list):
                print("🧟‍♂️ 이미 사망한 타겟. 포격 제외.")
                continue

        # 5. 포격 명령 추가
            latest_nearest_enemy = enemy
            retry_count = 0
            while retry_count < 3:
                try:
                    firing_cmds = turret.get_action_command(
                        player_data['pos'],
                        enemy,
                        turret_x_angle=player_data.get('turret_x', 0),
                        turret_y_angle=player_data.get('turret_y', 0),
                        player_y_angle=player_data.get('body_y', 0)
                    )
                    action_command += firing_cmds
                    print(f"🎯 포격 명령 추가됨: {enemy} (시도 {retry_count +1})")
                    
                    wait_for_impact_confirm(timeout=3.0)
                    if hit_state == 1:
                        print("✅ 명중 확인, 다음 타겟 진행")
                        break
                    else:
                        retry_count += 1
                        print(f"🔁 명중 실패, 재시도 {retry_count}/3")
                except ValueError as e:
                    print(f"🚫 포격 명령 생성 실패: {e}")
                    continue
            if retry_count >= 3:
                print("⚠️ 재시도 초과. 다음 타겟으로 이동")
                continue
            else:
                break

    return jsonify({"status": "OK", "message": "Rotation targeting sequence initiated."})

if __name__ == '__main__':
    threading.Thread(target=auto_start_rotation).start()
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)