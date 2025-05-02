from flask import Flask, request, jsonify, render_template
import logging
from flask_socketio import SocketIO
from PIL import Image, ImageDraw, ImageFont
import torch
from ultralytics import YOLO
import time
import json
import modules.turret as turret
import modules.is_near_enemy as is_near_enemy

app = Flask(__name__)

DEBUG = True

# response = {
#     'detections': filtered_results,
#     'nearest_enemy': nearest_enemy,
#     'fire_coordinates': fire_coordinates,
#     'control': 'continue'
# }


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
obstacles = []  # /set_obstacles 데이터 저장
latest_nearest_enemy = None

@app.route('/dashboard')
def dashboard():
    if DEBUG: print('🚨 dashboard >>>')
    return render_template('dashboard.html')

is_action_start = False
hit_state = -1
@app.route('/detect', methods=['POST'])
def detect():
    global player_data, obstacles, latest_nearest_enemy, action_command, destination, is_action_start, hit_state
    print('🌍 detect >>>')
    print('🤩🤩is_action_start', is_action_start)
    print('🤩🤩hit_state', is_action_start)


    # 1. 이미지 수신
    image = request.files.get('image')
    if not image:
        return jsonify([])

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
    class_colors = {
        'car002': 'red', 'car003': 'blue', 'car005': 'green', 'human001': 'orange',
        'rock001': 'purple', 'rock2': 'yellow', 'tank': 'cyan', 'wall001': 'pink', 'wall002': 'brown'
    }

    filtered_results = []
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", size=20)
    except:
        font = ImageFont.load_default()

    for box in detections:
        class_id = int(box[5])
        if class_id in target_classes: 
            filtered_results.append({
                'className': target_classes[class_id],
                'bbox': [float(coord) for coord in box[:4]],
                'confidence': float(box[4])
            })

    result_list = []
    for row in detections: 
        target_name = target_classes[row[-1]]
        target_bbox = [float(item) for item in row[:4]]
        confidence = float(row[-2])
        result_list.append({'className': target_name, 'bbox': target_bbox, 'confidence': confidence})
    
    # import pdb
    # pdb.set_trace()

    # 플레이어 위치
    player_pos = (
        player_data.get('pos', {}).get('x', 60),
        player_data.get('pos', {}).get('z', 57)
    )
    print("Player position:", player_pos)

    # 가장 가까운 적 찾기
    nearest_enemy = is_near_enemy.find_nearest_enemy(filtered_results, player_pos, obstacles)
    if nearest_enemy and not is_action_start:
        try:
            if DEBUG: print(f"👉 Generating action command: player_pos={player_data.get('pos')}, dest={destination}")
            latest_nearest_enemy = nearest_enemy
            action_command = turret.get_action_command(
                player_data.get('pos', {'x': 60, 'y': 10, 'z': 57}),
                nearest_enemy,
                turret_x_angle=player_data.get('turret_x', 0),
                turret_y_angle=player_data.get('turret_y', 0),
                player_y_angle=player_data.get('body_y', 0)
            )
            print('🐟action_command', action_command)
            is_action_start = True
        except ValueError as e:
            print(f"🚫 Error generating action command: {str(e)}")
            action_command = []
        
        print('🤩🤩action - is_action_start', is_action_start)
        print('🤩🤩action - hit_state', is_action_start)

    return jsonify(result_list)

@app.route('/info', methods=['POST'])
def info():
    global player_data
    data = request.get_json(force=True)
    if not data:
        if DEBUG: print("🚫 No JSON received")
        return jsonify({"error": "No JSON received", "control": ""}), 400

    player_data = {
        'pos': {
            'x': data.get('playerPos', {}).get('x', player_data.get('pos', {}).get('x', 60)),
            'y': data.get('playerPos', {}).get('y', player_data.get('pos', {}).get('y', 10)),
            'z': data.get('playerPos', {}).get('z', player_data.get('pos', {}).get('z', 57)),
        },
        'turret_x': data.get('playerTurretX', player_data.get('turret_x', 0)),
        'turret_y': data.get('playerTurretY', player_data.get('turret_y', 0)),
        'body_x': data.get('playerBodyX', player_data.get('body_x', 0)),
        'body_y': data.get('playerBodyY', player_data.get('body_y', 0)),
        'body_z': data.get('playerBodyZ', player_data.get('body_z', 0)),
    }
    if DEBUG: print(f"📍 Player data updated: {player_data}")
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
    global action_command, latest_nearest_enemy, is_action_start
    if DEBUG: print('🚨 get_action >>>', action_command)
    if action_command:
        command = action_command.pop(0)
        if DEBUG: print(f"🔫 Action Command: {command}")
        
        if hit_state == 1 and command['turret'] != 'FIRE' and command['weight'] == '0.0':
            # reverse 끝나는 지점
            is_action_start = False
            hit_state == -1
            # print("impact_control False", action_command)
            print('🤩🤩reverse end - is_action_start', is_action_start)
            print('🤩🤩reverse end - hit_state', is_action_start)

        return jsonify(command)
    else:
        return jsonify({"turret": "", "weight": 0.0})

@app.route('/update_bullet', methods=['POST'])
def update_bullet():
    global destination, impact_info, player_data, action_command, latest_nearest_enemy, hit_state
    if DEBUG: print('🚨 update_bullet >>>')
    data = request.get_json()
    action_command = []
    if not data:
        if DEBUG: print("🚫 Invalid bullet data")
        return jsonify({"status": "ERROR", "message": "Invalid request data"}), 400

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
        time.sleep(5)
        hit_state = 0
        try:
            action_command = turret.get_action_command(
                player_data.get('pos', {'x': 60, 'y': 10, 'z': 57}),
                latest_nearest_enemy,
                turret_x_angle=player_data.get('turret_x', 0),
                turret_y_angle=player_data.get('turret_y', 0),
                player_y_angle=player_data.get('body_y', 0)
            )
            if DEBUG: print('is_hit >> action_command:', action_command)
        except ValueError as e:
            if DEBUG: print(f"🚫 Error generating action command: {str(e)}")
            action_command = []
        
        print('🤩🤩re action - is_action_start', is_action_start)
        print('🤩🤩re action - hit_state', is_action_start)
    else:
        if DEBUG: print("Hit!!!!!")
        hit_state = 1
        action_command = turret.get_reverse_action_command(
            player_data.get('turret_x', 0),
            player_data.get('turret_y', 0),
            player_data.get('body_x', 0),
            player_data.get('body_y', 0),
        )
        
        print('🤩🤩reverse - is_action_start', is_action_start)
        print('🤩🤩reverse - hit_state', is_action_start)

    if DEBUG: print(f"💥 Bullet Impact at X={impact_info['x']}, Y={impact_info['y']}, Z={impact_info['z']}, Target={impact_info['target']}")

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
        return jsonify({'status': 'error', 'message': 'No data received'}), 400

    obstacles = data['obstacles']
    print(f"🪨 Obstacle data updated: {obstacles}")
    if DEBUG: print("Obstacle data:", obstacles)
    if DEBUG: print(f"🪨 Obstacle data updated: {json.dumps(obstacles, indent=2)}")
    return jsonify({'status': 'success', 'message': 'Obstacle data received'})

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