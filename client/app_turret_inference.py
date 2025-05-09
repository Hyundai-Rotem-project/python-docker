from flask import Flask, request, jsonify
import os
import torch
from ultralytics import YOLO
from datetime import datetime
import pandas as pd
import cv2
import numpy as np
import json


app = Flask(__name__)



time = 0
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


left_folder= "C:\\Users\\Dhan\\Documents\\Tank Challenge\\capture_images\\L"
right_folder = "C:\\Users\\Dhan\\Documents\\Tank Challenge\\capture_images\\R"
log_folder = "C:\\Users\\Dhan\\Documents\\Tank Challenge\\log_data\\tank_info_log.txt"

# ------ 3D 계산 함수 ------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('best.pt').to(device)


player_state = {"x": 0.0, "y": 0.0, "z": 0.0, "turret_yaw": 0.0, "turret_pitch": 0.0, "mode": "direct"}

reload_time = 7
last_fire_time = 0

range_table = {
    "direct": 50,
    "indirect": 120
}

HORIZONTAL_DEGREE_PER_WEIGHT = 21.35
VERTICAL_DEGREE_PER_WEIGHT = 2.67

def get_latest_log():
    log_files = sorted([f for f in os.listdir(log_folder) if f.endswith('.txt')])
    if not log_files:
        return None
    log_path = os.path.join(log_folder, log_files[-1])
    return pd.read_csv(log_path)

def calculate_disparity(left_img, right_img):
    gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    stereo = cv2.StereoSGBM_create(0, 64, 7, 8 * 3 * 7 ** 2, 32 * 3 * 7 ** 2, 1, 5, 50, 16)
    return stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0

def extract_3d_data(disparity, x, y, cx, cy, focal_length=1280.0, baseline=1):
    h, w = disparity.shape
    region = disparity[max(y-3, 0):min(y+3, h), max(x-3, 0):min(x+3, w)]
    valid = region[region > 0]

    if len(valid) == 0:
        return 0.0, 0.0, 0.0

    disp = float(np.median(valid))
    depth = (focal_length * baseline) / disp
    return round((x - cx) * depth / focal_length, 3), round((y - cy) * depth / focal_length, 3), round(depth, 3)

def get_detected_enemies(focal_length=1280.0):
    left_files = sorted(os.listdir(left_folder))
    right_files = sorted(os.listdir(right_folder))

    left_img = cv2.imread(os.path.join(left_folder, left_files[-1]))
    right_img = cv2.imread(os.path.join(right_folder, right_files[-1]))

    results = model(left_img)
    boxes = results[0].boxes
    disparity = calculate_disparity(left_img, right_img)
    
    print(results)

    h, w = left_img.shape[:2]
    cx, cy = w // 2, h // 2

    enemies = []

    for box in boxes:
        if float(box.conf[0]) < 0.3:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        class_id = int(box.cls.item())
        class_name = str(model.names[class_id])

        if class_name.lower() not in ["tank", "soldier"]:
            continue

        center_2d_x = (x1 + x2) // 2
        center_2d_y = (y1 + y2) // 2
        x3d, y3d, z3d = extract_3d_data(disparity, center_2d_x, center_2d_y, cx, cy)

        enemies.append({
            "class": class_name,
            "3d": np.array([x3d, y3d, z3d]),
            "confidence": round(float(box.conf[0]), 2)
        })

        test = {
            'x': x_test,
            'y': y_test,
            'z': z_test
        }

    return test





@app.route('/detect', methods=['POST'])
def detect():
    image = request.files.get('image')
    if not image:
        return jsonify({"error": "No image received"}), 400

    os.makedirs('./snapshot', exist_ok=True)
    global time
    image_path = f'./snapshot/image{time:.2f}.jpg'
    image_dir = "C:\\Users\\Dhan\\Documents\\Tank Challenge\\capture_images\\L"
    left_files = sorted(os.listdir(image_dir))
    img_path = os.path.join(image_dir, left_files[-1])
    #image_path='temp_image.jpg'
    image.save(image_path)

    results = model(img_path)
    detections = results[0].boxes.data.cpu().numpy()

    target_classes = {0: 'car2', 1: 'car3', 2: 'car5', 3: 'human1', 4: 'rock1', 5: 'rock2', 6: 'tank', 7: 'wall1', 8: 'wall2'}
    filtered_results = []
    for box in detections:
        class_id = int(box[5])
        if class_id in target_classes:
            filtered_results.append({
                'className': target_classes[class_id],
                'bbox': [float(coord) for coord in box[:4]],
                'confidence': float(box[4])
            })
    
    return jsonify(filtered_results)

@app.route('/info', methods=['POST'])
def info():
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON received"}), 400
    keys = [
    "time", "distance", "playerPos", "playerSpeed",
    "playerTurretX", "playerTurretY", "playerBodyX", "playerBodyY", "playerBodyZ",
    'stereoCameraLeftPos', 'stereoCameraLeftRot', 'stereoCameraRightPos', 'stereoCameraRightRot']
    new_keys=["playerBodyX", "playerBodyY", "playerBodyZ"]
    global time
    time = data['time']
    filtered_data = {key: data[key] for key in keys}

    output_data=get_detected_enemies()

    print("📨 /info data received:", output_data)


    # Auto-pause after 15 seconds
    #if data.get("time", 0) > 15:
    #    return jsonify({"status": "success", "control": "pause"})
    # Auto-reset after 15 seconds
    #if data.get("time", 0) > 15:
    #    return jsonify({"stsaatus": "success", "control": "reset"})
    return jsonify({"status": "success", "control": ""})

@app.route('/update_position', methods=['POST'])
def update_position():
    data = request.get_json()
    if not data or "position" not in data:
        return jsonify({"status": "ERROR", "message": "Missing position data"}), 400

    try:
        x, y, z = map(float, data["position"].split(","))
        current_position = (int(x), int(z))
        print(f"📍 Position updated: {current_position}")
        return jsonify({"status": "OK", "current_position": current_position})
    except Exception as e:
        return jsonify({"status": "ERROR", "message": str(e)}), 400

@app.route('/get_move', methods=['GET'])
def get_move():
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
    if action_command:
        command = action_command.pop(0)
        print(f"🔫 Action Command: {command}")
        return jsonify(command)
    else:
        return jsonify({"turret": "", "weight": 0.0})

@app.route('/update_bullet', methods=['POST'])
def update_bullet():
    data = request.get_json()
    if not data:
        return jsonify({"status": "ERROR", "message": "Invalid request data"}), 400

    print(f"💥 Bullet Impact at X={data.get('x')}, Y={data.get('y')}, Z={data.get('z')}, Target={data.get('hit')}")
    return jsonify({"status": "OK", "message": "Bullet impact data received"})

@app.route('/set_destination', methods=['POST'])
def set_destination():
    data = request.get_json()
    if not data or "destination" not in data:
        return jsonify({"status": "ERROR", "message": "Missing destination data"}), 400

    try:
        x, y, z = map(float, data["destination"].split(","))
        print(f"🎯 Destination set to: x={x}, y={y}, z={z}")
        return jsonify({"status": "OK", "destination": {"x": x, "y": y, "z": z}})
    except Exception as e:
        return jsonify({"status": "ERROR", "message": f"Invalid format: {str(e)}"}), 400

@app.route('/update_obstacle', methods=['POST'])
def update_obstacle():
    data = request.get_json()
    if not data:
        return jsonify({'status': 'error', 'message': 'No data received'}), 400

    print("🪨 Obstacle Data:", data)
    return jsonify({'status': 'success', 'message': 'Obstacle data received'})

#Endpoint called when the episode starts
@app.route('/init', methods=['GET'])
def init():
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
    print("🚀 /start command received")
    return jsonify({"control": ""})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

