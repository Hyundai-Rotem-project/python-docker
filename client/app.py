from flask import Flask, request, jsonify
import math
import torch
from ultralytics import YOLO
import time
import heapq
import asyncio
import threading
import requests
import os
import base64
from io import BytesIO
from PIL import Image
import numpy as np

app = Flask(__name__)

# Node 클래스
class Node:
    def __init__(self, x, z):
        self.x = x
        self.z = z
        self.is_obstacle = False

# Grid 클래스
class Grid:
    def __init__(self, width=300, height=300):
        self.width = width
        self.height = height
        self.grid = [[Node(x, z) for z in range(height)] for x in range(width)]

    def node_from_world_point(self, world_x, world_z):
        grid_x = max(0, min(int(world_x), self.width - 1))
        grid_z = max(0, min(int(world_z), self.height - 1))
        return self.grid[grid_x][grid_z]

    def set_obstacle(self, x_min, x_max, z_min, z_max):
        x_min = max(0, min(int(x_min), self.width - 1))
        x_max = max(0, min(int(x_max), self.width - 1))
        z_min = max(0, min(int(z_min), self.height - 1))
        z_max = max(0, min(int(z_max), self.height - 1))
        for x in range(x_min, x_max + 1):
            for z in range(z_min, z_max + 1):
                self.grid[x][z].is_obstacle = True
        print(f"🪨 Grid obstacle set: x_min={x_min}, x_max={x_max}, z_min={z_min}, z_max={z_max}")

    def get_neighbors(self, node):
        x, z = node.x, node.z
        neighbors = []
        for dx, dz in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, nz = x + dx, z + dz
            if 0 <= nx < self.width and 0 <= nz < self.height:
                if not self.grid[nx][nz].is_obstacle:
                    neighbors.append(self.grid[nx][nz])
        return neighbors

# 전역 변수
grid = Grid()
model = YOLO('best.pt')
obstacles = []
player_state = {
    "position": (60.0, 27.23),
    "last_position": None,
    "destination": None,
    "state": "IDLE",
    "distance_to_destination": float("inf"),
    "last_shot_time": 0.0,
    "shot_cooldown": 2.0,
    "body_x": 0.0,
    "body_y": 0.0,
    "body_z": 0.0,
    "last_valid_angle": None,
    "rotation_start_time": None,
    "pause_start_time": None,
    "enemy_detected": False,
    "last_shot_target": None
}
state_lock = threading.Lock()

# 상수
ROTATION_THRESHOLD_DEG = 5
STOP_DISTANCE = 45.0  # 목적지 도달 기준 거리 축소
SLOWDOWN_DISTANCE = 120.0
ROTATION_TIMEOUT = 0.8
PAUSE_DURATION = 0.5
WEIGHT_LEVELS = [1.0, 0.6, 0.3, 0.1, 0.05, 0.01]
DETECTION_RANGE = 100.0

# 클래스 정의
ENEMY_CLASSES = {'car2', 'car3', 'tank'}
FRIENDLY_CLASSES = {'car5'}
OBSTACLE_CLASSES = {'rock1', 'rock2', 'wall1', 'wall2'}

def select_weight(value, levels=WEIGHT_LEVELS):
    return min(levels, key=lambda x: abs(x - value))

def calculate_move_weight(distance):
    if distance <= STOP_DISTANCE:
        return 0.0
    elif distance > SLOWDOWN_DISTANCE:
        return 1.0
    normalized = (distance - STOP_DISTANCE) / (SLOWDOWN_DISTANCE - STOP_DISTANCE)
    target_weight = 0.01 + (1.0 - 0.01) * (normalized ** 2)
    return select_weight(target_weight)

def calculate_rotation_weight(angle_diff_deg):
    abs_deg = abs(angle_diff_deg)
    if abs_deg < ROTATION_THRESHOLD_DEG:
        return 0.0
    target_weight = min(0.3, abs_deg / 45)
    return select_weight(target_weight)

# YOLO 탐지로 장애물 분석
async def analyze_obstacle(obstacle, index):
    x_center = (obstacle["x_min"] + obstacle["x_max"]) / 2
    z_center = (obstacle["z_min"] + obstacle["z_max"]) / 2
    image_data = obstacle.get("image")

    target_classes = {0: 'car2', 1: 'car3', 2: 'car5', 3: 'human1', 4: 'rock1', 5: 'rock2', 6: 'tank', 7: 'wall1', 8: 'wall2'}
    if image_data:
        try:
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))
            results = model.predict(image, verbose=False)
            detections = results[0].boxes.data.cpu().numpy()
            filtered_results = []
            for box in detections:
                class_id = int(box[5])
                if class_id in target_classes:
                    filtered_results.append({
                        'className': target_classes[class_id],
                        'bbox': [float(coord) for coord in box[:4]],
                        'confidence': float(box[4])
                    })

            if filtered_results:
                detection = max(filtered_results, key=lambda x: x['confidence'])
                class_name = detection['className']
                confidence = detection['confidence']
                print(f"YOLO detection succeeded at ({x_center:.2f}, {z_center:.2f}): class={class_name}, confidence={confidence:.2f}")
            else:
                class_name = 'unknown'
                confidence = 0.0
                print(f"YOLO detection succeeded at ({x_center:.2f}, {z_center:.2f}): no valid detections")

            if class_name in ENEMY_CLASSES:
                print(f"Enemy detected: {class_name} at ({x_center:.2f}, {z_center:.2f})")
            elif class_name in FRIENDLY_CLASSES:
                print(f"Friendly detected: {class_name} at ({x_center:.2f}, {z_center:.2f})")
            elif class_name in OBSTACLE_CLASSES:
                print(f"Obstacle detected: {class_name} at ({x_center:.2f}, {z_center:.2f})")
            else:
                print(f"Unknown object detected: {class_name} at ({x_center:.2f}, {z_center:.2f})")

        except Exception as e:
            print(f"YOLO detection failed at ({x_center:.2f}, {z_center:.2f}): {e}")
            class_name = 'tank' if x_center < 80 else 'car5'
            if x_center > 90:
                class_name = 'rock1'
            print(f"Fallback detection: {class_name} at ({x_center:.2f}, {z_center:.2f})")
            if class_name in ENEMY_CLASSES:
                print(f"Enemy detected: {class_name} at ({x_center:.2f}, {z_center:.2f})")
            elif class_name in FRIENDLY_CLASSES:
                print(f"Friendly detected: {class_name} at ({x_center:.2f}, {z_center:.2f})")
            elif class_name in OBSTACLE_CLASSES:
                print(f"Obstacle detected: {class_name} at ({x_center:.2f}, {z_center:.2f})")
            else:
                print(f"Unknown object detected: {class_name} at ({x_center:.2f}, {z_center:.2f})")
    else:
        print(f"YOLO detection failed at ({x_center:.2f}, {z_center:.2f}): no image data provided")
        class_name = 'tank' if x_center < 80 else 'car5'
        if x_center > 90:
            class_name = 'rock1'
        print(f"Fallback detection: {class_name} at ({x_center:.2f}, {z_center:.2f})")
        if class_name in ENEMY_CLASSES:
            print(f"Enemy detected: {class_name} at ({x_center:.2f}, {z_center:.2f})")
        elif class_name in FRIENDLY_CLASSES:
            print(f"Friendly detected: {class_name} at ({x_center:.2f}, {z_center:.2f})")
        elif class_name in OBSTACLE_CLASSES:
            print(f"Obstacle detected: {class_name} at ({x_center:.2f}, {z_center:.2f})")
        else:
            print(f"Unknown object detected: {class_name} at ({x_center:.2f}, {z_center:.2f})")

    return {"className": class_name, "position": (x_center, z_center)}

# 포격 함수
async def shoot_at_target(target_pos):
    with state_lock:
        current_time = asyncio.get_event_loop().time()
        time_since_last_shot = current_time - player_state["last_shot_time"]
        if time_since_last_shot >= player_state["shot_cooldown"]:
            player_state["last_shot_time"] = current_time
            player_state["last_shot_target"] = target_pos
            player_state["enemy_detected"] = True
            bullet_data = {
                "x": target_pos[0],
                "y": 0.0,
                "z": target_pos[1],
                "hit": "enemy"
            }
            print(f"Attempting to shoot at {target_pos}, bullet: {bullet_data}, time_since_last_shot={time_since_last_shot:.2f}s")
            for attempt in range(2):
                try:
                    response = requests.post('http://localhost:5000/update_bullet', json=bullet_data, timeout=5)
                    print(f"Shot fired successfully at {target_pos}, status={response.status_code}")
                    return bullet_data
                except requests.RequestException as e:
                    print(f"Shot failed at {target_pos}: HTTP error on attempt {attempt+1}, error={e}")
            print(f"Shot failed at {target_pos}: all HTTP attempts failed")
            return None
        else:
            print(f"Shot failed at {target_pos}: cooldown active, {player_state['shot_cooldown'] - time_since_last_shot:.2f}s remaining")
            return None

# Grid 기반 A* 알고리즘
def a_star(start, goal, grid):
    def heuristic(node, goal_node):
        print(f"🛤️ Heuristic: node=({node.x}, {node.z}), goal_node=({goal_node.x}, {goal_node.z})")
        return math.sqrt((node.x - goal_node.x) ** 2 + (node.z - goal_node.z) ** 2)

    start_node = grid.node_from_world_point(start[0], start[1])
    goal_node = grid.node_from_world_point(goal[0], goal[1])
    print(f"🛤️ A* calculating path from ({start_node.x}, {start_node.z}) to ({goal_node.x}, {goal_node.z})")

    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start_node, goal_node), 0, start_node))
    came_from = {}
    g_score = {start_node: 0}
    f_score = {start_node: heuristic(start_node, goal_node)}

    while open_set:
        _, current_g, current = heapq.heappop(open_set)
        if current == goal_node:
            path = []
            while current in came_from:
                path.append((current.x, current.z))
                current = came_from[current]
            path.append((start_node.x, start_node.z))
            path.reverse()
            print(f"🛤️ A* path calculated: {path}")
            return path

        for neighbor in grid.get_neighbors(current):
            tentative_g_score = current_g + 1
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal_node)
                heapq.heappush(open_set, (f_score[neighbor], tentative_g_score, neighbor))
    print(f"🛤️ A* path calculation failed: no path from ({start_node.x}, {start_node.z}) to ({goal_node.x}, {goal_node.z})")
    return []

# 비동기 이동 및 장애물 처리
async def move_towards_destination():
    global obstacles, grid
    print("🚀 Starting move_towards_destination")
    while player_state["destination"] and player_state["state"] not in ["STOPPED", "IDLE"]:
        with state_lock:
            current_pos = player_state["position"]
            dest = player_state["destination"]
        distance_to_dest = math.sqrt((dest[0] - current_pos[0])**2 + (dest[1] - current_pos[1])**2)
        print(f"🚗 Current position: {current_pos}, destination: {dest}, distance: {distance_to_dest:.2f}, state: {player_state['state']}")

        if distance_to_dest < STOP_DISTANCE:
            with state_lock:
                player_state["position"] = dest
                player_state["state"] = "STOPPED"
            print(f"🎯 Reached destination: {dest}")
            break

        path = a_star(current_pos, dest, grid)
        if not path:
            with state_lock:
                player_state["state"] = "STOPPED"
            print("🚫 Stopping: no valid A* path to destination")
            break

        next_pos = path[1] if len(path) > 1 else dest
        distance = math.sqrt((next_pos[0] - current_pos[0])**2 + (next_pos[1] - current_pos[1])**2)
        print(f"🚗 Moving to waypoint: {next_pos}, current position: {current_pos}, distance: {distance:.2f}")
        
        if distance < 1.0:
            with state_lock:
                player_state["position"] = next_pos
            print(f"✅ Reached waypoint: {next_pos}")
            continue

        for idx, obstacle in enumerate(obstacles):
            obs_center = ((obstacle["x_min"] + obstacle["x_max"]) / 2, (obstacle["z_min"] + obstacle["z_max"]) / 2)
            obs_distance = math.sqrt((obs_center[0] - current_pos[0])**2 + (obs_center[1] - current_pos[1])**2)
            if obs_distance < DETECTION_RANGE:
                detection = await analyze_obstacle(obstacle, idx)
                class_name = detection["className"]
                if class_name in ENEMY_CLASSES:
                    await shoot_at_target(obs_center)
                    await asyncio.sleep(0.5)

        await asyncio.sleep(0.1)
    print("🏁 move_towards_destination stopped")

# 비동기 태스크를 스레드에서 실행
def run_async_task():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(move_towards_destination())
    loop.close()

@app.route('/detect', methods=['POST'])
def detect():
    image = request.files.get('image')
    if not image:
        print("YOLO detection failed: no image received in /detect")
        return jsonify({"error": "No image received"}), 400

    try:
        image = Image.open(image)
        results = model.predict(image, verbose=False)
        detections = results[0].boxes.data.cpu().numpy()

        target_classes = {0: 'car2', 1: 'car3', 2: 'car5', 3: 'human1', 4: 'rock1', 5: 'rock2', 6: 'tank', 7: 'wall1', 8: 'wall2'}
        filtered_results = []
        for box in detections:
            class_id = int(box[5])
            if class_id in target_classes:
                class_name = target_classes[class_id]
                filtered_results.append({
                    'className': class_name,
                    'bbox': [float(coord) for coord in box[:4]],
                    'confidence': float(box[4])
                })
                print(f"YOLO detection succeeded in /detect: class={class_name}, confidence={float(box[4]):.2f}")
                if class_name in ENEMY_CLASSES:
                    print(f"Enemy detected in /detect: {class_name}")
                elif class_name in FRIENDLY_CLASSES:
                    print(f"Friendly detected in /detect: {class_name}")
                elif class_name in OBSTACLE_CLASSES:
                    print(f"Obstacle detected in /detect: {class_name}")
                else:
                    print(f"Unknown object detected in /detect: {class_name}")
        if not filtered_results:
            print("YOLO detection succeeded in /detect: no valid detections")
        return jsonify(filtered_results)
    except Exception as e:
        print(f"YOLO detection failed in /detect: {e}")
        return jsonify({"error": "Detection failed"}), 500

@app.route('/info', methods=['POST'])
def info():
    global player_state
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON received"}), 400

    with state_lock:
        player_pos = data.get("playerPos")
        player_state["body_x"] = data.get("playerBodyX", 0.0)
        player_state["body_y"] = data.get("playerBodyY", 0.0)
        player_state["body_z"] = data.get("playerBodyZ", 0.0)
        player_state["distance_to_destination"] = data.get("distance", float("inf"))
        player_state["position"] = (player_pos["x"], player_pos["z"])

    if not player_state["destination"]:
        with state_lock:
            player_state["state"] = "IDLE"
        return jsonify({"status": "success", "control": "STOP", "weight": 0.0})

    current_angle = math.radians(player_state["body_x"])
    with state_lock:
        if player_state["last_position"] and player_state["position"] != player_state["last_position"]:
            dx = player_state["position"][0] - player_state["last_position"][0]
            dz = player_state["position"][1] - player_state["last_position"][1]
            if math.sqrt(dx**2 + dz**2) > 0.0001:
                current_angle = math.atan2(dz, dx)
                player_state["last_valid_angle"] = current_angle

        if player_state["last_valid_angle"] is None and player_state["destination"]:
            dx, dz = player_state["destination"]
            px, pz = player_state["position"]
            player_state["last_valid_angle"] = math.atan2(dz - pz, dx - px)

    control = "STOP"
    weight = 0.0
    current_time = time.time()

    with state_lock:
        if player_state["state"] == "IDLE" and player_state["destination"]:
            player_state["state"] = "ROTATING"
            player_state["rotation_start_time"] = current_time

        elif player_state["state"] == "ROTATING":
            dx, dz = player_state["destination"]
            px, pz = player_state["position"]
            target_angle = math.atan2(dz - pz, dx - px)
            angle_diff = ((target_angle - current_angle + math.pi) % (2 * math.pi)) - math.pi
            angle_deg = math.degrees(angle_diff)

            if current_time - player_state.get("rotation_start_time", 0) > ROTATION_TIMEOUT:
                player_state["state"] = "PAUSE"
                player_state["pause_start_time"] = current_time
            elif abs(angle_deg) < ROTATION_THRESHOLD_DEG:
                player_state["state"] = "PAUSE"
                player_state["pause_start_time"] = current_time
            else:
                control = "D" if angle_diff > 0 else "A"
                weight = calculate_rotation_weight(angle_deg)

        elif player_state["state"] == "PAUSE":
            if current_time - player_state["pause_start_time"] >= PAUSE_DURATION:
                player_state["state"] = "MOVING"
                control = "W"
                weight = calculate_move_weight(player_state["distance_to_destination"])
                threading.Thread(target=run_async_task, daemon=True).start()
            else:
                control = "STOP"
                weight = 0.0

        elif player_state["state"] == "MOVING":
            dx, dz = player_state["destination"]
            px, pz = player_state["position"]
            angle_diff = ((math.atan2(dz - pz, dx - px) - current_angle + math.pi) % (2 * math.pi)) - math.pi
            angle_deg = math.degrees(angle_diff)

            if abs(angle_deg) > ROTATION_THRESHOLD_DEG * 6:
                player_state["state"] = "ROTATING"
                player_state["rotation_start_time"] = current_time
                control = "D" if angle_diff > 0 else "A"
                weight = calculate_rotation_weight(angle_deg)
            else:
                control = "W"
                weight = calculate_move_weight(player_state["distance_to_destination"])

        elif player_state["state"] == "STOPPED":
            control = "STOP"
            weight = 0.0

        player_state["last_position"] = player_state["position"]

    return jsonify({"status": "success", "control": control, "weight": weight})

@app.route('/update_position', methods=['POST'])
def update_position():
    global player_state
    data = request.get_json()
    if not data or "position" not in data:
        return jsonify({"status": "ERROR", "message": "Missing position data"}), 400

    try:
        x, y, z = map(float, data["position"].split(","))
        with state_lock:
            player_state["position"] = (x, z)
        return jsonify({"status": "OK", "current_position": player_state["position"]})
    except Exception as e:
        return jsonify({"status": "ERROR", "message": str(e)}), 400

@app.route('/get_move', methods=['GET'])
def get_move():
    with state_lock:
        if player_state["state"] == "STOPPED":
            return jsonify({"move": "STOP", "weight": 0.0})
        elif player_state["state"] == "MOVING":
            weight = calculate_move_weight(player_state["distance_to_destination"])
            return jsonify({"move": "W", "weight": weight})
        elif player_state["state"] == "ROTATING":
            return jsonify({"move": "A", "weight": 0.3})
        elif player_state["state"] == "PAUSE":
            return jsonify({"move": "STOP", "weight": 0.0})
        else:
            return jsonify({"move": "STOP", "weight": 0.0})

@app.route('/get_action', methods=['GET'])
def get_action():
    with state_lock:
        if player_state["enemy_detected"] or player_state["state"] == "STOPPED":
            player_state["enemy_detected"] = False
            print(f"🔫 /get_action: Returning FIRE, target={player_state['last_shot_target']}")
            return jsonify({"turret": "FIRE", "weight": 1.0})
        else:
            print("🔫 /get_action: No enemy detected, no action")
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
    global player_state
    data = request.get_json()
    if not data or "destination" not in data:
        return jsonify({"status": "ERROR", "message": "Missing destination data"}), 400

    try:
        x, y, z = map(float, data["destination"].split(","))
        if not (0 <= x < grid.width and 0 <= z < grid.height):
            return jsonify({"status": "ERROR", "message": f"Destination ({x}, {z}) out of grid bounds (0-{grid.width}, 0-{grid.height})"}), 400
        with state_lock:
            player_state["destination"] = (x, z)
            player_state["state"] = "ROTATING"
            player_state["rotation_start_time"] = time.time()
        print(f"🎯 Destination set to: ({x}, {z})")
        return jsonify({"status": "OK", "destination": {"x": x, "y": y, "z": z}})
    except ValueError as e:
        return jsonify({"status": "ERROR", "message": f"Invalid format: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"status": "ERROR", "message": f"Error: {str(e)}"}), 400

@app.route('/update_obstacle', methods=['POST'])
def update_obstacle():
    global obstacles, grid
    data = request.get_json()
    if not data:
        return jsonify({'status': 'error', 'message': 'No data received'}), 400

    obstacles = data.get('obstacles', [])
    grid = Grid()
    for obstacle in obstacles:
        detection = asyncio.run(analyze_obstacle(obstacle, 0))
        if detection["className"] in OBSTACLE_CLASSES:
            grid.set_obstacle(obstacle["x_min"], obstacle["x_max"], obstacle["z_min"], obstacle["z_max"])
    print("🪨 Obstacle Data:", obstacles)
    return jsonify({'status': 'success', 'message': 'Obstacle data received'})

@app.route('/init', methods=['GET'])
def init():
    config = {
        "startMode": "start",
        "blStartX": 60,
        "blStartY": 10,
        "blStartZ": 27.23,
        "rdStartX": 59,
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