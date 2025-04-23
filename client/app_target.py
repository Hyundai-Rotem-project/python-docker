from flask import Flask, request, jsonify
import os
import torch
from ultralytics import YOLO
import heapq
import math
import time

app = Flask(__name__)
model = YOLO('yolov8n.pt')

def rotate_vector(vector, angle_deg):
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    x, z = vector
    return (
        x * cos_a - z * sin_a,
        x * sin_a + z * cos_a
    )

def normalize(vec):
    length = math.hypot(vec[0], vec[1])
    if length == 0:
        return (0, 0)
    return (vec[0]/length, vec[1]/length)

# 상태 저장
latest_position = {"x": 0, "z": 0, "yaw": 0}
destination = None
obstacles = set()

# 맵 경계
MAP_WIDTH, MAP_HEIGHT = 300, 300

last_goal_update_time = 0

# # Move commands with weights (11+ variations)
move_command = [
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
latest_position = None
latest_body_vector = None  # 방향 벡터 저장

@app.route('/info', methods=['POST'])
def info():
    global latest_info, latest_body_vector
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON received"}), 400
    latest_info = data
    print("📨 /info data received:", data)
    try:
        latest_body_vector = (
            float(data.get("playerBodyX", 0)),
            float(data.get("playerBodyZ", 0))
        )
    except Exception as e:
        print("⚠️ 바디 벡터 파싱 오류:", str(e))
        latest_body_vector = (0.0, 1.0)  # 기본값: 북쪽

    return jsonify({"status": "success", "control": ""})

def angle_between_vectors(v1, v2):
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
    if mag1 == 0 or mag2 == 0:
        return 0
    cos_theta = dot / (mag1 * mag2)
    cos_theta = max(-1.0, min(1.0, cos_theta))  # 안정화
    angle = math.acos(cos_theta)
    return math.degrees(angle)

def get_direction_command(current, next_pos):
    dx = next_pos[0] - current[0]
    dz = next_pos[1] - current[1]

    distance = math.sqrt(dx**2 + dz**2)
    if distance < 10:  # 너무 가까우면 멈춤
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
    global destination, move_command, latest_body_vector, latest_position

    data = request.get_json()
    if not data or "destination" not in data:
        return jsonify({"status": "ERROR", "message": "Missing destination data"}), 400

    try:
        x, y, z = map(float, data["destination"].split(","))
        destination = (x, z)
        goal_position = (int(round(x)), int(round(z)))
        print(f"🎯 목적지 설정: {goal_position}")

        if latest_position is None or latest_body_vector is None:
            return jsonify({"status": "WAITING", "message": "Waiting for position or direction data"}), 202
        
        start_position = (int(round(latest_position[0])), int(round(latest_position[1])))
        path = a_star(start_position, goal_position)

        # 현재 방향 보정용 단일 step 계산
        temp_path = a_star(start_position, goal_position)
        if not temp_path:
            move_command.append({"move": "STOP", "weight": 1.0})
            return jsonify({"status": "OK", "destination": {"x": x, "y": y, "z": z}})

        first_step = temp_path[0]
        direction_vec = normalize(latest_body_vector)
        target_vec = (first_step[0] - start_position[0], first_step[1] - start_position[1])
        angle = angle_between_vectors(direction_vec, target_vec)
        cross = direction_vec[0]*target_vec[1] - direction_vec[1]*target_vec[0]

        # 회전 -> 정지 -> 회전이 끝났는지 확인 후 -> 회전
        if angle > 15:
            rotate_cmd = "A" if cross > 0 else "D"
            move_command.append({"move": rotate_cmd, "weight": 1.0})
            move_command.append({"move": "STOP", "weight": 1.0})

            # 방향 보정 이후, 새 방향으로 업데이트
            direction_vec = normalize(target_vec)       

        # 보정된 방향으로 A* 재계산
        path = a_star(start_position, goal_position)
        if not path:
            move_command.append({"move": "STOP", "weight": 1.0})
            return jsonify({"status": "OK", "destination": {"x": x, "y": y, "z": z}})

        current = start_position
        for step in path:
            target_vec = (step[0] - current[0], step[1] - current[1])
            angle = angle_between_vectors(direction_vec, target_vec)
            cross = direction_vec[0]*target_vec[1] - direction_vec[1]*target_vec[0]

            if angle > 15:
                rotate_cmd = "A" if cross > 0 else "D"
                move_command.append({"move": rotate_cmd, "weight": 1.0})
                move_command.append({"move": "STOP", "weight": 1.0})
                direction_vec = normalize(target_vec)

            direction = get_direction_command(current, step)
            move_command.append({"move": direction, "weight": 1.0})
            move_command.append({"move": "STOP", "weight": 1.0})
            current = step

        move_command.append({"move": "STOP", "weight": 1.0})

        return jsonify({"status": "OK", "destination": {"x": x, "y": y, "z": z}})
     
    except Exception as e:
        return jsonify({"status": "ERROR", "message": f"Invalid format: {str(e)}"}), 400

obstacles = set()

def is_within_map_bounds(pos):
    x, z = pos
    return 0 <= x <= MAP_WIDTH and 0 <= z <= MAP_HEIGHT

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])  # 맨해튼 거리

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
        
            tentative_g = g_score[current] + 1
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    print("🚫 경로 없음")
    return []  # 경로 없음

def path_to_commands(path, initial_yaw):
    commands = []
    yaw = initial_yaw

    for i in range(1, len(path)):
        prev = path[i-1]
        curr = path[i]

        dx = curr[0] - prev[0]
        dz = curr[1] - prev[1]

        target_yaw = math.degrees(math.atan2(dz, dx))
        angle_diff = (target_yaw - yaw + 360) % 360
        if angle_diff > 180:
            angle_diff -= 360

        # 회전 보정
        if abs(angle_diff) > 5:
            commands.append({"action": "rotate", "angle": angle_diff})
            yaw = (yaw + angle_diff) % 360

        # 전진
        commands.append({"action": "move", "distance": 1})
    
    return commands

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
        # 아군 현재 위치 초기화
    latest_position = (config["blStartX"], config["blStartZ"])

    print(f"🔵 아군 시작 위치: {latest_position}")
    print(f"🔴 적군 위치: ({config['rdStartX']}, {config['rdStartZ']})")

    return jsonify(config)

@app.route('/start', methods=['GET'])
def start():
    print("🚀 /start command received")
    return jsonify({"control": ""})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
