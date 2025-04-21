from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO
import requests
import os
import torch
from ultralytics import YOLO
import time
import modules.turret as turret

app = Flask(__name__)
model = YOLO('yolov8n.pt')
socketio = SocketIO(app)

# Move commands with weights (11+ variations)
move_command = []

# Action commands with weights (15+ variations)
action_command = []

PLAYER_DATA = {}
DESTINATION = ()

@app.route('/dashboard')
def dashboard():
    print('🚨 dashboard >>>')
    return render_template('dashboard.html')


@app.route('/detect', methods=['POST'])
def detect():
    print('🚨 detect >>>')
    """Receives an image from the simulator, performs object detection, and returns filtered results."""
    # 1. 이미지 받기 및 저장
    image = request.files.get('image')
    if not image:
        return jsonify({"error": "No image received"}), 400

    image_path = 'temp_image.jpg'
    image.save(image_path)

    # 2. YOLO 모델 처리
    results = model(image_path)
    detections = results[0].boxes.data.cpu().numpy()


    # 3. 결과 필터링 및 변환
    target_classes = {0: "person", 2: "car", 7: "truck", 15: "rock"}
    filtered_results = []
    for box in detections:
        class_id = int(box[5])
        if class_id in target_classes:
            filtered_results.append({
                'className': target_classes[class_id],
                'bbox': [float(coord) for coord in box[:4]],
                'confidence': float(box[4])
            })

    response = jsonify(filtered_results)
    return response

@app.route('/info', methods=['POST'])
def info():
    print('🚨 info >>>')
    global PLAYER_DATA
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON received"}), 400

    # print("📨 /info data received:", data)
    
    PLAYER_DATA = {
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
    

    # Auto-pause after 15 seconds
    #if data.get("time", 0) > 15:
    #    return jsonify({"status": "success", "control": "pause"})
    # Auto-reset after 15 seconds
    #if data.get("time", 0) > 15:
    #    return jsonify({"stsaatus": "success", "control": "reset"})
    return jsonify({"status": "success", "control": ""})

@app.route('/update_position', methods=['POST'])
def update_position():
    print('🚨 update_position >>>')
    data = request.get_json()
    if not data or "position" not in data:
        return jsonify({"status": "ERROR", "message": "Missing position data"}), 400

    try:
        x, y, z = map(float, data["position"].split(","))
        current_position = (int(x), int(y), int(z))
        print(f"📍 Position updated: {current_position}")
        return jsonify({"status": "OK", "current_position": current_position})
    except Exception as e:
        return jsonify({"status": "ERROR", "message": str(e)}), 400

@app.route('/get_move', methods=['GET'])
def get_move():
    print('🚨 get_move >>>')
    global move_command
    if move_command:
        command = move_command.pop(0)
        print(f"🚗 Move Command: {command}")
        return jsonify(command)
    else:
        return jsonify({"move": "STOP", "weight": 1.0})

@app.route('/get_action', methods=['GET'])
def get_action():
    global action_command
    print('🚨 get_action >>>', action_command)
    if action_command:
        command = action_command.pop(0)
        print(f"🔫 Action Command: {command}")
        return jsonify(command)
    else:
        return jsonify({"turret": "", "weight": 0.0})

@app.route('/update_bullet', methods=['POST'])
def update_bullet():
    print('🚨 update_bullet >>>')
    data = request.get_json()
    if not data:
        return jsonify({"status": "ERROR", "message": "Invalid request data"}), 400

    impact_info = {
        'x': data.get('x'),
        'y': data.get('y'),
        'z': data.get('z'),
        'target': data.get('hit'),  # 'terrain' 또는 다른 타겟 정보
        'timestamp': time.strftime('%H:%M:%S')
    }
    
    print(f"💥 Bullet Impact at X={impact_info['x']}, Y={impact_info['y']}, Z={impact_info['z']}, Target={impact_info['target']}")
    
    socketio.emit('bullet_impact', impact_info)
    
    return jsonify({"status": "OK", "message": "Bullet impact data received"})

@app.route('/set_destination', methods=['POST'])
def set_destination():
    print('🚨 set_destination >>>')
    global DESTINATION
    global action_command
    data = request.get_json()
    action_command = []
    if not data or "destination" not in data:
        return jsonify({"status": "ERROR", "message": "Missing destination data"}), 400

    try:
        x, y, z = map(float, data["destination"].split(","))
        DESTINATION = {
            'x': x,
            'y': y,
            'z': z,
        }
        print(f"🎯 Destination set to: x={x}, y={y}, z={z}")
        action_command = turret.generate_action_command(PLAYER_DATA['pos'], PLAYER_DATA['turret_x'], PLAYER_DATA['turret_y'], DESTINATION)
        print('action_command????', action_command)
        return jsonify({"status": "OK", "destination": {"x": x, "y": y, "z": z}})
    except Exception as e:
        return jsonify({"status": "ERROR", "message": f"Invalid format: {str(e)}"}), 400

@app.route('/update_obstacle', methods=['POST'])
def update_obstacle():
    print('🚨 update_obstacle >>>')
    data = request.get_json()
    if not data:
        return jsonify({'status': 'error', 'message': 'No data received'}), 400

    print("🪨 Obstacle Data:", data)
    return jsonify({'status': 'success', 'message': 'Obstacle data received'})

#Endpoint called when the episode starts
@app.route('/init', methods=['GET'])
def init():
    print('🚨 init >>>')
    config = {
        "startMode": "start",  # Options: "start" or "pause"
        "blStartX": 60,  #Blue Start Position
        "blStartY": 10,
        "blStartZ": 27.23,
        "rdStartX": 59, #Red Start Position
        "rdStartY": 10,
        "rdStartZ": 280
    }
    print("🛠️ Initialization config sent via /init:", config)
    return jsonify(config)

@app.route('/start', methods=['GET'])
def start():
    print('🚨 start >>>')
    # print("🚀 /start command received")
    return jsonify({"control": ""})

@app.route('/test_rotation', methods=['POST'])
def test_rotation():
    global action_command
    data = request.get_json()
    rotation_type = data.get('type', 'Q')  # Q, E, F, R
    count = data.get('count', 1)  # 회전 명령 횟수
    
    # 기존 명령어 초기화 후 새로운 테스트 명령 추가
    action_command = []  # 기존 명령어 초기화
    
    # 회전 명령 추가 (각 명령 사이에 정지 명령 추가)
    for _ in range(count):
        action_command.append({"turret": rotation_type, "weight": 0.5})
    action_command.append({"turret": rotation_type, "weight": 0.0})  # 각 회전 후 정지

    test_info = {
        'rotation_type': rotation_type,
        'count': count,
        'timestamp': time.strftime('%H:%M:%S'),
        'rotation_desc': {
            'Q': 'Left',
            'E': 'Right',
            'F': 'Down',
            'R': 'Up'
        }.get(rotation_type, 'Unknown')
    }
    
    print(f"🔄 Testing {test_info['rotation_desc']} rotation ({rotation_type}) x {count}")
    socketio.emit('rotation_test', test_info)
    print("action_command >>", action_command)
    
    return jsonify({"status": "OK", "message": "Rotation test started"})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
