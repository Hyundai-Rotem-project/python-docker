from flask import Flask, request, jsonify
import os
import torch
from ultralytics import YOLO
import heapq
import math
import time

app = Flask(__name__)
model = YOLO('yolov8n.pt')

fixed_goal = None
last_goal_update_time = 0

# # Move commands with weights (11+ variations)
move_command = [
    {"move": "W", "weight": 1.0},
    {"move": "W", "weight": 1.0},
    {"move": "W", "weight": 1.0},
    {"move": "W", "weight": 1.0},
    {"move": "W", "weight": 1.0},
    {"move": "W", "weight": 1.0},
    {"move": "W", "weight": 1.0},
    {"move": "W", "weight": 1.0},
    {"move": "W", "weight": 1.0},
    {"move": "W", "weight": 1.0},
    {"move": "W", "weight": 1.0},
    {"move": "W", "weight": 1.0},
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

@app.route('/detect', methods=['POST'])
def detect():
    image = request.files.get('image')
    if not image:
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
                'confidence': float(box[4])
            })

    return jsonify(filtered_results)

latest_info = {}  # /info에서 받은 데이터 저장

@app.route('/info', methods=['POST'])
def info():
    global latest_info
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON received"}), 400
    latest_info = data
    print("📨 /info data received:", data)
    return jsonify({"status": "success", "control": ""})

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])  # 맨해튼 거리

# 경계선에 가까운지 확인하고, 패널티를 부여하는 함수
def penalty_near_boundary(x, z, reference_pos, margin=50, penalty_weight=5.0):
    """탱크의 기준 위치(reference_pos)에서 얼마나 멀리 떨어졌는지에 따라 경계 회피 가중치 부여"""
    dx = abs(x - reference_pos[0])
    dz = abs(z - reference_pos[1])

    # 일정 거리 이상 벗어났으면 경계라고 가정하고 패널티 줌
    if dx > margin or dz > margin:
        penalty = (dx + dz - margin) * penalty_weight
        return penalty
    return 0  # 기준 위치 근처는 패널티 없음

def a_star(start, goal, grid_size=1):
    global obstacles
    if goal in obstacles:
        print(f"⚠️ 목표 지점 {goal}은 장애물입니다!")
        return []
    
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current) 
                current = came_from[current]
            path.reverse()
            print(f"🛤️ A* 경로: {path}")
            return path

        for dx, dz in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            neighbor = (current[0] + dx * grid_size, current[1] + dz * grid_size)
            if neighbor in obstacles:
                continue
            
            # 경계 패널티 계산
            move_cost = 1.414 if dx != 0 and dz != 0 else 1
            boundary_penalty = penalty_near_boundary(neighbor[0], neighbor[1], start)
            tentative_g_score = g_score[current] + 1 + boundary_penalty  # 경계 회피 패널티 추가

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return []  # 경로 없음

def get_direction_command(current, next_pos):
    dx = next_pos[0] - current[0]
    dz = next_pos[1] - current[1]

    distance = math.sqrt(dx**2 + dz**2)
    if distance < 0.1:  # 너무 가까우면 멈춤
        print(f"🛑 너무 가까움: 현재={current}, 다음={next_pos}")
        return "STOP"
    
    dx /= distance
    dz /= distance

    # 👇 방향 판단 후 명령 반환, 대각선 이동
    print(f"🧭 방향 계산: 현재={current}, 다음={next_pos}, dx={dx}, dz={dz}")
    if abs(dz) > abs(dx) and dz > 0:
        return "W"
    elif abs(dz) > abs(dx) and dz < 0:
        return "S"
    elif abs(dx) > abs(dz) and dx > 0:
        return "D"
    elif abs(dx) > abs(dz) and dx < 0:
        return "A"
    return "STOP"

@app.route('/update_position', methods=['POST'])
def update_position():
    global latest_position
    data = request.get_json()
    print(f"📍 위치 업데이트: {latest_position}")

    if not data or "position" not in data:
        return jsonify({"status": "ERROR", "message": "Missing position data"}), 400

    try:
        x, y, z = map(float, data["position"].split(","))
        latest_position = (x,z)  # 🧭 실시간 위치 저장
        print(f"📍 Position updated: {latest_position}")
        return jsonify({"status": "OK", "current_position": latest_position})
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

obstacles = set()

@app.route('/update_obstacle', methods=['POST'])
def update_obstacle():
    global obstacles
    data = request.get_json()
    if not data or 'obstacles' not in data:
        return jsonify({'status': 'error', 'message': 'No obstacle data received'}), 400

    obstacles = set((int(round(obs['x'])), int(round(obs['z']))) for obs in data['obstacles'])
    print("🪨 Obstacle Data:", data)
    print("🪨 Updated obstacles:", obstacles)
    return jsonify({'status': 'success', 'message': 'Obstacle data updated successfully'})

#Endpoint called when the episode starts
latest_info = {}  # /info에서 받은 데이터 저장
latest_position = None

@app.route('/init', methods=['GET'])
def init():
    global latest_info

    config = {
        "startMode": "start",  # Options: "start" or "pause"
        "blStartX": 60,  #Blue Start Position
        "blStartY": 10,
        "blStartZ": 27.23,
        "rdStartX": 59, #Red Start Position
        "rdStartY": 10,
        "rdStartZ": 280,
    }

        # 👉 아군/적군 위치 저장
    latest_info = {
        "playerPos": {
            "x": config["blStartX"],
            "y": config["blStartY"],
            "z": config["blStartZ"]
        },
        "enemyPos": {
            "x": config["rdStartX"],
            "y": config["rdStartY"],
            "z": config["rdStartZ"]
        }
    }
    print("🛠️ Initialization config sent via /init:", config)
    return jsonify(config)

@app.route('/start', methods=['GET'])
def start():
    print("🚀 /start command received")
    return jsonify({"control": ""})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
