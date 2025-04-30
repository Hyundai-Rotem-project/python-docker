from flask import Flask, request, jsonify
import os
import torch
from ultralytics import YOLO
import numpy as np
import math
import logging
import json

app = Flask(__name__)
model = YOLO('yolov8n.pt')

# 로깅 설정
logging.basicConfig(filename='tank.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# 전역 변수
player_position = None  # 아군 전차 위치 (x, z)
obstacles = []  # 장애물 리스트

# Move commands with weights (11+ variations)
move_command = [
    {"move": "W", "weight": 1.0},
    {"move": "W", "weight": 0.6},
    {"move": "W", "weight": 0.3},
    {"move": "D", "weight": 1.0},
    {"move": "D", "weight": 0.6},
    {"move": "D", "weight": 0.4},
    {"move": "A", "weight": 1.0},
    {"move": "A", "weight": 0.3},
    {"move": "S", "weight": 0.5},
    {"move": "S", "weight": 0.1},
    {"move": "STOP"}
]

# Action commands with weights (15+ variations)
action_command = [
    {"turret": "Q", "weight": 1.0},
    {"turret": "Q", "weight": 0.8},
    {"turret": "Q", "weight": 0.6},
    {"turret": "Q", "weight": 0.4},
    {"turret": "E", "weight": 1.0},
    {"turret": "E", "weight": 1.0},
    {"turret": "E", "weight": 1.0},
    {"turret": "E", "weight": 1.0},
    {"turret": "F", "weight": 0.5},
    {"turret": "F", "weight": 0.3},
    {"turret": "R", "weight": 1.0},
    {"turret": "R", "weight": 0.7},
    {"turret": "R", "weight": 0.4},
    {"turret": "R", "weight": 0.2},
    {"turret": "FIRE"}
]

def find_nearest_obstacle_to_enemy(detections, player_pos, obstacles):
    """YOLO 탐지된 적 중에서 플레이어와 가장 가까운 적과 장애물 반환"""
    enemy_classes = {'car', 'truck'}
    enemies = [det for det in detections if det['className'] in enemy_classes and det['confidence'] >= 0.3]
    
    if not enemies:
        logging.info("No enemy detected")
        print("⚠️ No enemy detected")
        return {"nearest_enemy": {"message": "No enemy detected"}, "nearest_obstacle": {"message": "No enemy detected"}}
    
    if not player_pos:
        logging.error("Player position not set")
        # print("⚠️ Player position not set")
        return {"nearest_enemy": {"message": "Player position not set"}, "nearest_obstacle": {"message": "Player position not set"}}
    
    # 적 위치 가정: 임의로 enemyPos 사용 (실제로는 YOLO bbox 변환 필요)
    enemy_pos = {'x': 59.22119903564453, 'z': 279.630615234375}  # 임시 값
    nearest_enemy = {
        'className': enemies[0]['className'],  # 첫 번째 적 선택 (단순화)
        'x': enemy_pos['x'],
        'z': enemy_pos['z'],
        'distance': math.sqrt((enemy_pos['x'] - player_pos[0])**2 + (enemy_pos['z'] - player_pos[1])**2)
    }
    
    # 가장 가까운 적 정보 출력
    enemy_log = (
        f"Nearest enemy: class={nearest_enemy['className']}, "
        f"x={nearest_enemy['x']:.6f}, z={nearest_enemy['z']:.6f}, "
        f"distance={nearest_enemy['distance']:.2f}m"
    )
    logging.info(enemy_log)
    print(f"🚀 {enemy_log}")
    
    if not obstacles:
        logging.info("No obstacles available")
        print("⚠️ No obstacles available")
        return {"nearest_enemy": nearest_enemy, "nearest_obstacle": {"message": "No obstacles available"}}
    
    min_distance = float('inf')
    nearest_obstacle = None
    
    for obs in obstacles:
        # 장애물 중심점 계산
        obs_center_x = (obs['x_min'] + obs['x_max']) / 2
        obs_center_z = (obs['z_min'] + obs['z_max']) / 2
        # 플레이어와 장애물 중심 간 거리
        distance = math.sqrt((obs_center_x - player_pos[0])**2 + (obs_center_z - player_pos[1])**2)
        if distance < min_distance:
            min_distance = distance
            nearest_obstacle = {
                'x_min': obs['x_min'],
                'x_max': obs['x_max'],
                'z_min': obs['z_min'],
                'z_max': obs['z_max']
            }
    
    obstacle_log = (
        f"Nearest obstacle: x_min={nearest_obstacle['x_min']:.6f}, "
        f"x_max={nearest_obstacle['x_max']:.6f}, z_min={nearest_obstacle['z_min']:.6f}, "
        f"z_max={nearest_obstacle['z_max']:.6f}, distance={min_distance:.2f}m"
    )
    logging.info(obstacle_log)
    # print(f"🚀 {obstacle_log}")
    
    return {"nearest_enemy": nearest_enemy, "nearest_obstacle": nearest_obstacle}

@app.route('/detect', methods=['POST'])
def detect():
    global player_position, obstacles
    image = request.files.get('image')
    if not image:
        logging.error("No image received")
        # print("🚫 No image received")
        return jsonify({"error": "No image received"}), 400

    image_path = 'temp_image.jpg'
    image.save(image_path)

    results = model(image_path)
    detections = results[0].boxes.data.cpu().numpy()

    target_classes = {0: "person", 2: "car", 7: "truck", 15: "rock"}
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

    # 가장 가까운 적과 장애물 찾기
    result = find_nearest_obstacle_to_enemy(filtered_results, player_position, obstacles)
    
    # 좌표 출력
    nearest_enemy = result['nearest_enemy']
    nearest_obstacle = result['nearest_obstacle']

    if 'message' in nearest_enemy:
        enemy_log = f"Nearest enemy: {nearest_enemy['message']}"
    else:
        enemy_log = (
            f"Nearest enemy coordinates: x={nearest_enemy['x']:.6f}, "
            f"z={nearest_enemy['z']:.6f}"
        )
    if 'message' in nearest_obstacle:
        obstacle_log = f"Nearest obstacle: {nearest_obstacle['message']}"
    else:
        obstacle_log = (
            f"Nearest obstacle coordinates: x_min={nearest_obstacle['x_min']:.6f}, "
            f"x_max={nearest_obstacle['x_max']:.6f}, z_min={nearest_obstacle['z_min']:.6f}, "
            f"z_max={nearest_obstacle['z_max']:.6f}"
        )
    
    logging.info(enemy_log)
    logging.info(obstacle_log)
    print(f"🚀 {enemy_log}")
    print(f"🚀 {obstacle_log}")

    response = {
        'detections': filtered_results,
        'nearest_enemy': result['nearest_enemy'],
        'nearest_obstacle': result['nearest_obstacle']
    }
    logging.info(f"Detection response: {json.dumps(response, indent=2)}")
    print(f"Detection response: {json.dumps(response, indent=2)}")
    return jsonify(response)

@app.route('/info', methods=['POST'])
def info():
    data = request.get_json(force=True)
    test = data.copy()
    test.pop('lidarPoints', None)
    print('❤️❤️', test)
    if not data:
        logging.error("No JSON received")
        print("🚫 No JSON received")
        return jsonify({"error": "No JSON received"}), 400

    # # 디버깅: 수신된 데이터 전체 출력
    # logging.info(f"Received /info data: {json.dumps(data, indent=2)}")
    # # print(f"Received /info data: {json.dumps(data, indent=2)}")

    response = {"status": "success", "control": ""}
    return jsonify(response)

@app.route('/update_position', methods=['POST'])
def update_position():
    global player_position
    data = request.get_json()
    if not data or "position" not in data:
        logging.error("Missing position data")
        print("🚫 Missing position data")
        return jsonify({"status": "ERROR", "message": "Missing position data"}), 400

    try:
        x, y, z = map(float, data["position"].split(","))
        player_position = (x, z)
        logging.info(f"Position updated: {player_position}")
        # print(f"📍 Position updated: {player_position}")
        return jsonify({"status": "OK", "current_position": player_position})
    except Exception as e:
        logging.error(f"Invalid position format: {str(e)}")
        # print(f"🚫 Invalid position format: {str(e)}")
        return jsonify({"status": "ERROR", "message": str(e)}), 400

@app.route('/update_obstacle', methods=['POST'])
def update_obstacle():
    global obstacles
    data = request.get_json()
    if not data or 'obstacles' not in data:
        logging.error("No obstacle data received")
        print("🚫 No obstacle data received")
        return jsonify({'status': 'error', 'message': 'No data received'}), 400

    obstacles = data['obstacles']
    logging.info(f"Obstacle data updated: {obstacles}")
    print(f"🪨 Obstacle data updated: {obstacles}")
    return jsonify({'status': 'success', 'message': 'Obstacle data received'})

@app.route('/get_move', methods=['GET'])
def get_move():
    global move_command
    if move_command:
        command = move_command.pop(0)
        logging.info(f"Move Command: {command}")
        print(f"🚗 Move Command: {command}")
        return jsonify(command)
    else:
        return jsonify({"move": "STOP", "weight": 1.0})

@app.route('/get_action', methods=['GET'])
def get_action():
    global action_command
    if action_command:
        command = action_command.pop(0)
        logging.info(f"Action Command: {command}")
        print(f"🔫 Action Command: {command}")
        return jsonify(command)
    else:
        return jsonify({"turret": "", "weight": 0.0})

@app.route('/update_bullet', methods=['POST'])
def update_bullet():
    data = request.get_json()
    if not data:
        logging.error("Invalid bullet data")
        print("🚫 Invalid bullet data")
        return jsonify({"status": "ERROR", "message": "Invalid request data"}), 400

    logging.info(f"Bullet Impact at X={data.get('x')}, Y={data.get('y')}, Z={data.get('z')}, Target={data.get('hit')}")
    print(f"💥 Bullet Impact at X={data.get('x')}, Y={data.get('y')}, Z={data.get('z')}, Target={data.get('hit')}")
    return jsonify({"status": "OK", "message": "Bullet impact data received"})

@app.route('/set_destination', methods=['POST'])
def set_destination():
    data = request.get_json()
    if not data or "destination" not in data:
        logging.error("Missing destination data")
        print("🚫 Missing destination data")
        return jsonify({"status": "ERROR", "message": "Missing destination data"}), 400

    try:
        x, y, z = map(float, data["destination"].split(","))
        logging.info(f"Destination set to: x={x}, y={y}, z={z}")
        print(f"🎯 Destination set to: x={x}, y={y}, z={z}")
        return jsonify({"status": "OK", "destination": {"x": x, "y": y, "z": z}})
    except Exception as e:
        logging.error(f"Invalid destination format: {str(e)}")
        print(f"🚫 Invalid destination format: {str(e)}")
        return jsonify({"status": "ERROR", "message": f"Invalid format: {str(e)}"}), 400

@app.route('/collision', methods=['POST']) 
def collision():
    data = request.get_json()
    if not data:
        logging.error("No collision data received")
        # print("🚫 No collision data received")
        return jsonify({'status': 'error', 'message': 'No collision data received'}), 400

    object_name = data.get('objectName')
    position = data.get('position', {})
    x = position.get('x')
    y = position.get('y')
    z = position.get('z')
    logging.info(f"Collision Detected - Object: {object_name}, Position: ({x}, {y}, {z})")
    # print(f"💥 Collision Detected - Object: {object_name}, Position: ({x}, {y}, {z})")
    return jsonify({'status': 'success', 'message': 'Collision data received'})

@app.route('/init', methods=['GET'])
def init():
    config = {
        "startMode": "start",
        "blStartX": 60,
        "blStartY": 10,
        "blStartZ": 27.23,
        "rdStartX": 59,
        "rdStartY": 10,
        "rdStartZ": 280,
        "trackingMode": True,
        "detactMode": False,
        "logMode": True,
        "enemyTracking": True,
        "saveSnapshot": False,
        "saveLog": True,
        "saveLidarData": False,
        "lux": 30000
    }
    logging.info(f"Initialization config sent: {config}")
    # print(f"🛠️ Initialization config sent via /init: {config}")
    return jsonify(config)

@app.route('/start', methods=['GET'])
def start():
    logging.info("Start command received")
    print("🚀 /start command received")
    return jsonify({"control": ""})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)