import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import torch
torch.set_num_threads(1)
from flask import Flask, request, jsonify
import math
import time
import heapq
import asyncio
import threading
import requests
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import logging
from logging.handlers import RotatingFileHandler
from ultralytics import YOLO

app = Flask(__name__)

# 로깅 설정
handler = RotatingFileHandler('tank.log', maxBytes=10*1024*1024, backupCount=5)
logging.basicConfig(handlers=[handler], level=logging.INFO, format='%(asctime)s - %(message)s')

# YOLO 모델 로드
try:
    model = YOLO(os.getenv('YOLO_MODEL_PATH', 'best.pt'))
except Exception as e:
    logging.error(f"YOLO model load failed: {e}")
    raise

# 상수
TARGET_CLASSES = {0: 'car2', 1: 'car3', 2: 'car5', 3: 'human1', 4: 'rock1', 5: 'rock2', 6: 'tank', 7: 'wall1', 8: 'wall2'}
PAUSE_DURATION = 0.5  # 요청 간격(0.7초)에 맞게 조정
SHOT_COOLDOWN = 0.7   # 요청 간격과 동기화
FIRING_RANGE = 45.0
CONFIDENCE_THRESHOLD = 0.7  # 오탐지 감소
ROTATION_THRESHOLD_DEG = 10
ROTATION_TIMEOUT = 2.0
WEIGHT_LEVELS = [1.0, 0.6, 0.3, 0.1, 0.05, 0.01]
DETECTION_RANGE = 100.0
ENEMY_CLASSES = {'car2', 'car3', 'tank'}
FRIENDLY_CLASSES = {'car5'}
OBSTACLE_CLASSES = {'rock1', 'rock2', 'wall1', 'wall2', 'human1'}
TRAPPED_TIMEOUT = 1.0
ESCAPE_ROTATION_ANGLE = 90.0
SHOTS_PER_ENEMY = 1
SHOT_INTERVAL = 0.7

# Node 클래스
class Node:
    def __init__(self, x, z):
        self.x = x
        self.z = z
        self.is_obstacle = False
        self.g = float('inf')
        self.h = 0.0

    def __lt__(self, other):
        return (self.g + self.h) < (other.g + other.h)

    def __eq__(self, other):
        return isinstance(other, Node) and self.x == other.x and self.z == other.z

    def __hash__(self):
        return hash((self.x, self.z))

    def __repr__(self):
        return f"Node(x={self.x}, z={self.z}, g={self.g:.2f}, h={self.h:.2f})"

# Grid 클래스
class Grid:
    def __init__(self, width=300, height=300):
        self.width = width
        self.height = height
        self.grid = [[Node(x, z) for z in range(height)] for x in range(width)]

    def clamp_coord(self, coord, max_val):
        return max(0, min(int(round(coord)), max_val - 1))

    def node_from_world_point(self, world_x, world_z):
        grid_x = self.clamp_coord(world_x, self.width)
        grid_z = self.clamp_coord(world_z, self.height)
        return self.grid[grid_x][grid_z]

    def set_obstacle(self, x_min, x_max, z_min, z_max, start_pos=None, goal_pos=None):
        x_min = self.clamp_coord(x_min, self.width)
        x_max = self.clamp_coord(x_max, self.width)
        z_min = self.clamp_coord(z_min, self.height)
        z_max = self.clamp_coord(z_max, self.height)
        for x in range(x_min, x_max + 1):
            for z in range(z_min, z_max + 1):
                if start_pos and goal_pos:
                    if (x, z) == (int(round(start_pos[0])), int(round(start_pos[1]))) or \
                       (x, z) == (int(round(goal_pos[0])), int(round(goal_pos[1]))):
                        continue
                self.grid[x][z].is_obstacle = True
        logging.info(f"Obstacle set: x_min={x_min}, x_max={x_max}, z_min={z_min}, z_max={z_max}")

    def get_neighbors(self, node):
        x, z = node.x, node.z
        neighbors = []
        for dx, dz in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nx, nz = x + dx, z + dz
            if 0 <= nx < self.width and 0 <= nz < self.height and not self.grid[nx][nz].is_obstacle:
                neighbors.append(self.grid[nx][nz])
        return neighbors

# 전역 변수
grid = Grid()
obstacles = []
player_state = {
    "position": (60.0, 27.23),
    "waypoints": [],
    "last_position": None,
    "destination": None,
    "state": "IDLE",
    "distance_to_destination": float("inf"),
    "last_shot_time": 0.0,
    "shot_cooldown": SHOT_COOLDOWN,
    "body_x": 0.0,
    "body_y": 0.0,
    "body_z": 0.0,
    "rotation_start_time": None,
    "pause_start_time": None,
    "enemy_detected": False,
    "last_shot_target": None,
    "last_body_x": None,
    "last_move_time": None,
    "last_move_position": None,
    "escape_rotation_target": None,
    "escape_start_time": None,
    "shots_fired": 0
}
state_lock = threading.Lock()

# 유틸리티 함수
def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def select_weight(value, levels=WEIGHT_LEVELS):
    return min(levels, key=lambda x: abs(x - value))

def calculate_move_weight(distance):
    return select_weight(min(1.0, distance / 100.0))

def calculate_rotation_weight(angle_diff_deg):
    abs_deg = abs(angle_diff_deg)
    if abs_deg < ROTATION_THRESHOLD_DEG:
        return 0.0
    target_weight = min(0.3, abs_deg / 90)
    return select_weight(target_weight)

def calculate_min_rotation(current_angle, target_angle):
    return (target_angle - current_angle + 180) % 360 - 180

def calculate_target_angle(current_pos, target_pos):
    dx = target_pos[0] - current_pos[0]
    dz = target_pos[1] - current_pos[1]
    return math.degrees(math.atan2(dz, dx)) % 360

async def analyze_obstacle(obstacle, index):
    x_center = (obstacle["x_min"] + obstacle["x_max"]) / 2
    z_center = (obstacle["z_min"] + obstacle["z_max"]) / 2
    image_data = obstacle.get("image")
    class_name = 'unknown'
    confidence = 0.0

    if not image_data:
        logging.info(f"YOLO: No image, classified as unknown at ({x_center:.2f}, {z_center:.2f})")
        return {"className": class_name, "position": (x_center, z_center), "confidence": confidence}

    try:
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        start_time = time.time()
        results = model.predict(image, verbose=False, conf=CONFIDENCE_THRESHOLD)
        logging.info(f"YOLO prediction took {time.time() - start_time:.2f} seconds")
        detections = results[0].boxes.data.cpu().numpy()
        filtered_results = [
            {
                'className': TARGET_CLASSES[int(box[5])],
                'confidence': float(box[4]),
                'bbox': [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
            }
            for box in detections if int(box[5]) in TARGET_CLASSES
        ]
        if filtered_results:
            detection = max(filtered_results, key=lambda x: x['confidence'])
            class_name = detection['className']
            confidence = detection['confidence']
            logging.info(f"YOLO: {class_name} at ({x_center:.2f}, {z_center:.2f}), confidence={confidence:.2f}")
        else:
            logging.info(f"YOLO: No valid detections at ({x_center:.2f}, {z_center:.2f})")
    except Exception as e:
        logging.error(f"YOLO failed: {e}")

    if class_name in ENEMY_CLASSES and confidence >= CONFIDENCE_THRESHOLD:
        logging.info(f"Enemy detected: {class_name} at ({x_center:.2f}, {z_center:.2f}), confidence={confidence:.2f}")
    elif class_name in FRIENDLY_CLASSES:
        logging.info(f"Friendly detected: {class_name} at ({x_center:.2f}, {z_center:.2f}), confidence={confidence:.2f}")
    elif class_name in OBSTACLE_CLASSES:
        logging.info(f"Obstacle detected: {class_name} at ({x_center:.2f}, {z_center:.2f}), confidence={confidence:.2f}")
    else:
        logging.info(f"Unknown object at ({x_center:.2f}, {z_center:.2f}), class={class_name}")

    return {"className": class_name, "position": (x_center, z_center), "confidence": confidence}

async def shoot_at_target(target_pos, target_type="enemy"):
    with state_lock:
        current_time = asyncio.get_event_loop().time()
        if current_time - player_state["last_shot_time"] < player_state["shot_cooldown"]:
            logging.info(f"Shot skipped: Cooldown active, remaining {player_state['shot_cooldown'] - (current_time - player_state['last_shot_time']):.2f}s")
            return None
        player_state["last_shot_time"] = current_time
        player_state["last_shot_target"] = target_pos
        bullet_data = {"x": target_pos[0], "y": 0.0, "z": target_pos[1], "hit": target_type}
        for attempt in range(2):
            try:
                response = requests.post('http://localhost:5000/update_bullet', json=bullet_data, timeout=0.7)
                if response.status_code == 200:
                    player_state["shots_fired"] += 1
                    logging.info(f"Shot #{player_state['shots_fired']} fired at {target_type} {target_pos}")
                    return bullet_data
                else:
                    logging.error(f"Shot failed: HTTP {response.status_code}, response={response.text}")
            except requests.RequestException as e:
                logging.error(f"Shot failed, attempt {attempt+1}: {e}")
        logging.error("Shot failed: All attempts failed")
        return None

def a_star(start, goal, grid):
    def heuristic(node, goal_node):
        return distance((node.x, node.z), (goal_node.x, goal_node.z))

    if not (0 <= start[0] < grid.width and 0 <= start[1] < grid.height):
        logging.error(f"A* failed: Invalid start {start}")
        return []
    if not (0 <= goal[0] < grid.width and 0 <= goal[1] < grid.height):
        logging.error(f"A* failed: Invalid goal {goal}")
        return []

    start_node = grid.node_from_world_point(start[0], start[1])
    goal_node = grid.node_from_world_point(goal[0], goal[1])

    if start_node.is_obstacle:
        logging.error(f"A* failed: Start is obstacle at ({start_node.x}, {start_node.z})")
        return []
    if goal_node.is_obstacle:
        logging.error(f"A* failed: Goal is obstacle at ({goal_node.x}, {goal_node.z})")
        return []

    open_set = []
    counter = 0
    heapq.heappush(open_set, (0 + heuristic(start_node, goal_node), 0, counter, start_node))
    came_from = {}
    g_score = {start_node: 0}
    f_score = {start_node: heuristic(start_node, goal_node)}

    while open_set:
        _, current_g, _, current = heapq.heappop(open_set)  # heappush -> heappop 수정
        if current == goal_node:
            path = []
            while current in came_from:
                path.append((current.x, current.z))
                current = came_from[current]
            path.append((start_node.x, start_node.z))
            path.reverse()
            return path
        for neighbor in grid.get_neighbors(current):
            tentative_g_score = g_score[current] + (1.4 if abs(neighbor.x - current.x) + abs(neighbor.z - current.z) > 1 else 1)
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal_node)
                counter += 1
                heapq.heappush(open_set, (f_score[neighbor], tentative_g_score, counter, neighbor))
    logging.error(f"A* failed: No path from ({start_node.x}, {start_node.z}) to ({goal_node.x}, {goal_node.z})")
    return []

def pure_pursuit_target(current_pos, path, lookahead_distance=5.0):
    if not path:
        logging.info("Pure pursuit: Empty path, returning current position")
        return current_pos
    x0, z0 = current_pos
    while len(path) > 1:
        px, pz = path[0]
        if distance((px, pz), (x0, z0)) < lookahead_distance:
            path.pop(0)
        else:
            break
    return path[0]

def move_to_firing_range(current_pos, enemy_pos, firing_range=FIRING_RANGE, grid=grid):
    dist = distance(current_pos, enemy_pos)
    if dist <= firing_range:
        logging.info(f"Already in firing range: {dist:.2f}m")
        return current_pos
    direction = [(enemy_pos[0] - current_pos[0]) / dist, (enemy_pos[1] - current_pos[1]) / dist]
    target_pos = (
        current_pos[0] + direction[0] * (dist - firing_range),
        current_pos[1] + direction[1] * (dist - firing_range)
    )
    path = a_star(current_pos, target_pos, grid)
    if not path:
        logging.warning(f"No path to firing range: {target_pos}")
        move_dist = min(1.0, dist - firing_range)
        next_pos = (
            current_pos[0] + direction[0] * move_dist,
            current_pos[1] + direction[1] * move_dist
        )
        if not check_obstacle_collision(*next_pos, obstacles):
            return next_pos
        return current_pos
    next_pos = path[1] if len(path) > 1 else target_pos
    logging.info(f"Moving to firing range ({firing_range}m): {next_pos}")
    return next_pos

async def move_towards_destination():
    global obstacles, grid
    logging.info("Starting move_towards_destination")
    attempt = 0
    max_attempts = 3
    while True:
        with state_lock:
            if not player_state["destination"] or player_state["state"] in ["STOPPED", "IDLE"]:
                break
            current_pos = player_state["position"]
            current_angle = player_state["body_x"]
            dest = player_state["destination"]
            state = player_state["state"]
            current_time = asyncio.get_event_loop().time()
            if player_state["last_move_time"] is None:
                player_state["last_move_time"] = current_time
                player_state["last_move_position"] = current_pos
        distance = distance(dest, current_pos)
        logging.info(f"Pos={current_pos}, Dest={dest}, Distance={distance:.2f}, State={state}, BodyX={current_angle:.2f}")

        if distance <= 1.0:
            with state_lock:
                player_state["state"] = "STOPPED"
                player_state["destination"] = None
                player_state["last_move_time"] = None
                player_state["last_move_position"] = None
            logging.info(f"Arrived at destination: {dest}")
            break

        if state not in ["MOVING", "ESCAPING", "PAUSE", "ROTATING"]:
            await asyncio.sleep(0.2)
            continue

        if state == "MOVING":
            move_distance = distance(current_pos, player_state["last_move_position"])
            if current_time - player_state["last_move_time"] > TRAPPED_TIMEOUT and move_distance < 0.5:
                with state_lock:
                    player_state["state"] = "ESCAPING"
                    player_state["escape_rotation_target"] = (current_angle + ESCAPE_ROTATION_ANGLE) % 360
                    player_state["escape_start_time"] = current_time
                    player_state["last_move_time"] = current_time
                    player_state["last_move_position"] = current_pos
                logging.info(f"Trapped detected: No movement for {TRAPPED_TIMEOUT}s at {current_pos}")
                logging.info(f"Starting ESCAPING: Rotating to {player_state['escape_rotation_target']:.2f}°")
                await asyncio.sleep(0.2)
                continue

        target_detected = False
        logging.info(f"Checking {len(obstacles)} obstacles")
        for idx, obstacle in enumerate(obstacles):
            obs_center = ((obstacle["x_min"] + obstacle["x_max"]) / 2, (obstacle["z_min"] + obstacle["z_max"]) / 2)
            obs_distance = distance(obs_center, current_pos)
            if obs_distance < DETECTION_RANGE:
                detection = await analyze_obstacle(obstacle, idx)
                if detection["className"] in ENEMY_CLASSES and detection["confidence"] >= CONFIDENCE_THRESHOLD:
                    target_detected = True
                    with state_lock:
                        player_state["state"] = "ROTATING"
                        player_state["rotation_start_time"] = current_time
                        player_state["enemy_detected"] = True
                        player_state["last_shot_target"] = obs_center
                        player_state["shots_fired"] = 0
                    target_angle = calculate_target_angle(current_pos, obs_center)
                    delta_angle = calculate_min_rotation(current_angle, target_angle)
                    control = "A" if delta_angle > 0 else "D"
                    weight = calculate_rotation_weight(abs(delta_angle))
                    logging.info(f"Enemy detected at {obs_center}, distance={obs_distance:.2f}m, rotating to {target_angle:.2f}°")
                    if abs(delta_angle) < ROTATION_THRESHOLD_DEG:
                        logging.info(f"Rotation complete: body_x={current_angle:.2f}, target={target_angle:.2f}")
                        new_pos = move_to_firing_range(current_pos, obs_center)
                        with state_lock:
                            player_state["position"] = new_pos
                            player_state["last_move_time"] = current_time
                            player_state["last_move_position"] = new_pos
                        await shoot_at_target(obs_center, target_type="enemy")
                        logging.info(f"Fired at enemy at {obs_center}, moving to destination")
                        with state_lock:
                            grid = Grid()
                            start_pos = player_state["position"]
                            goal_pos = player_state["destination"]
                            for obs in obstacles:
                                obs_detection = asyncio.run(analyze_obstacle(obs, 0))
                                if obs_detection["className"] in OBSTACLE_CLASSES and obs_detection["confidence"] >= CONFIDENCE_THRESHOLD:
                                    grid.set_obstacle(
                                        obs["x_min"], obs["x_max"],
                                        obs["z_min"], obs["z_max"],
                                        start_pos=start_pos, goal_pos=goal_pos
                                    )
                        target_angle = calculate_target_angle(new_pos, dest)
                        delta_angle = calculate_min_rotation(current_angle, target_angle)
                        control = "A" if delta_angle > 0 else "D"
                        weight = calculate_rotation_weight(abs(delta_angle))
                        if abs(delta_angle) < ROTATION_THRESHOLD_DEG:
                            player_state["state"] = "MOVING"
                            control = "W"
                            weight = calculate_move_weight(distance)
                        break
                else:
                    logging.info(f"Skipped shooting: Not an enemy or low confidence ({detection['className']}, confidence={detection['confidence']:.2f})")

        if target_detected:
            await asyncio.sleep(PAUSE_DURATION)
            continue

        path = a_star(current_pos, dest, grid)
        if not path:
            logging.error(f"A* failed, attempt {attempt+1}/{max_attempts}")
            attempt += 1
            if attempt >= max_attempts:
                dx, dz = dest[0] - current_pos[0], dest[1] - current_pos[1]
                dist = distance(dest, current_pos)
                if dist > 1.0 and dist > 1e-6:
                    move_dist = min(1.0, dist)
                    next_x = current_pos[0] + (dx / dist) * move_dist
                    next_z = current_pos[1] + (dz / dist) * move_dist
                    if not check_obstacle_collision(next_x, next_z, obstacles):
                        with state_lock:
                            player_state["position"] = (next_x, next_z)
                            player_state["last_move_time"] = current_time
                            player_state["last_move_position"] = (next_x, next_z)
                        logging.info(f"Fallback: Moved to ({next_x:.2f}, {next_z:.2f})")
                    else:
                        logging.error(f"Fallback: Collision at ({next_x:.2f}, {next_z:.2f})")
                        with state_lock:
                            player_state["state"] = "STOPPED"
                            player_state["destination"] = None
                            player_state["last_move_time"] = None
                            player_state["last_move_position"] = None
                        break
                else:
                    with state_lock:
                        player_state["state"] = "STOPPED"
                        player_state["destination"] = None
                        player_state["last_move_time"] = None
                        player_state["last_move_position"] = None
                    logging.error("Max attempts reached or too close")
                    break
                await asyncio.sleep(1.0)
            continue

        attempt = 0
        next_pos = path[1] if len(path) > 1 else dest
        dx = next_pos[0] - current_pos[0]
        dz = next_pos[1] - current_pos[1]
        dist = distance(next_pos, current_pos)
        if dist > 1e-6:
            angle_to_next = math.degrees(math.atan2(dz, dx)) % 360
            delta_angle = calculate_min_rotation(current_angle, angle_to_next)
            if abs(delta_angle) > 135:
                logging.warning(f"Skipping waypoint {next_pos}: delta_angle={delta_angle:.2f}°, too far from body_x={current_angle:.2f}")
                player_state["state"] = "ROTATING"
                player_state["rotation_start_time"] = current_time
                control = "A" if delta_angle > 0 else "D"
                weight = calculate_rotation_weight(abs(delta_angle))
                await asyncio.sleep(0.2)
                continue
        with state_lock:
            player_state["position"] = next_pos
            player_state["last_move_time"] = current_time
            player_state["last_move_position"] = next_pos
        logging.info(f"Moved to waypoint: {next_pos}, angle_to_next={angle_to_next:.2f}°")
        await asyncio.sleep(0.2)
    logging.info("Movement stopped")

def check_obstacle_collision(x, z, obstacles):
    for obs in obstacles:
        if obs["x_min"] <= x <= obs["x_max"] and obs["z_min"] <= z <= obs["z_max"]:
            logging.error(f"Collision at ({x:.2f}, {z:.2f})")
            return True
    return False

async def run_async_task():
    from flask import current_app
    with current_app.app_context():
        await move_towards_destination()

@app.route('/detect', methods=['POST'])
def detect():
    image = request.files.get('image')
    if not image:
        logging.error("YOLO: No image")
        return jsonify({"error": "No image received"}), 400
    try:
        image = Image.open(image).convert('RGB')
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        results = model.predict(image, verbose=False, conf=CONFIDENCE_THRESHOLD)
        detections = results[0].boxes.data.cpu().numpy()
        filtered_results = [
            {
                'className': TARGET_CLASSES[int(box[5])],
                'bbox': [float(coord) for coord in box[:4]],
                'confidence': float(box[4]),
                'x_min': float(box[0]),
                'x_max': float(box[2]),
                'z_min': float(box[1]),
                'z_max': float(box[3]),
                'image': image_base64
            }
            for box in detections if int(box[5]) in TARGET_CLASSES
        ]
        global obstacles
        obstacles = [
            {
                'x_min': res['x_min'],
                'x_max': res['x_max'],
                'z_min': res['z_min'],
                'z_max': res['z_max'],
                'image': res['image']
            }
            for res in filtered_results
        ]
        logging.info(f"YOLO: {len(filtered_results)} detections, updated obstacles")
        return jsonify(filtered_results)
    except Exception as e:
        logging.error(f"YOLO failed: {e}")
        return jsonify({"error": "Detection failed"}), 500

@app.route('/info', methods=['POST'])
def info():
    data = request.get_json(force=True)
    if not data:
        logging.error("No JSON received")
        return jsonify({"error": "No JSON received"}), 400

    with state_lock:
        player_pos = data.get("playerPos")
        body_x = data.get("playerBodyX", 0.0)
        body_x = body_x % 360 if body_x >= 0 else (body_x % 360 + 360)
        player_state["body_x"] = body_x
        player_state["body_y"] = data.get("playerBodyY", 0.0)
        player_state["body_z"] = data.get("playerBodyZ", 0.0)
        player_state["distance_to_destination"] = data.get("distance", float("inf"))
        player_state["position"] = (player_pos["x"], player_pos["z"])
        if player_state["last_body_x"] is not None:
            dbx = body_x - player_state["last_body_x"]
            if abs(dbx) < 1e-3 and player_state["state"] in ["ROTATING", "ESCAPING"]:
                current_time = time.time()
                if player_state["rotation_start_time"] and current_time - player_state["rotation_start_time"] > 0.5:
                    target = player_state.get("escape_rotation_target", body_x)
                    player_state["body_x"] = target
                    logging.warning(f"Forced body_x update to {player_state['body_x']:.2f} after delay, expected={target:.2f}")
                logging.warning(f"bodyX change too small during {player_state['state']}: ΔbodyX={dbx:.3f}, body_x={body_x:.2f}")
            logging.info(f"ΔbodyX={dbx:.3f}")
        player_state["last_body_x"] = body_x

    if not player_state["destination"]:
        with state_lock:
            player_state["state"] = "IDLE"
            player_state["last_move_time"] = None
            player_state["last_move_position"] = None
        logging.info("IDLE: No destination")
        return jsonify({"status": "success", "control": "STOP", "weight": 0.0})

    current_angle = player_state["body_x"]
    control = "STOP"
    weight = 0.0
    expected_body_x = None
    current_time = time.time()

    with state_lock:
        if player_state["state"] == "IDLE" and player_state["destination"]:
            player_state["state"] = "ROTATING"
            player_state["rotation_start_time"] = current_time
            logging.info("IDLE -> ROTATING")

        elif player_state["state"] == "ROTATING":
            target_angle = calculate_target_angle(player_state["position"], player_state["destination"])
            delta_angle = calculate_min_rotation(current_angle, target_angle)
            expected_body_x = target_angle
            control = "A" if delta_angle > 0 else "D"
            angle_diff_deg = abs(delta_angle)
            logging.info(f"ROTATING: angle_diff={angle_diff_deg:.2f}°, control={control}, expected_body_x={expected_body_x:.2f}")
            if abs(delta_angle) < ROTATION_THRESHOLD_DEG:
                player_state["state"] = "PAUSE"
                player_state["pause_start_time"] = current_time
                logging.info(f"ROTATING -> PAUSE: body_x={player_state['body_x']:.2f}, expected_body_x={expected_body_x:.2f}")
            elif player_state["rotation_start_time"] and (current_time - player_state["rotation_start_time"]) > ROTATION_TIMEOUT:
                player_state["state"] = "PAUSE"
                player_state["pause_start_time"] = current_time
                logging.info("ROTATING -> PAUSE: timeout")
            else:
                weight = calculate_rotation_weight(angle_diff_deg)

        elif player_state["state"] == "ESCAPING":
            expected_body_x = player_state["escape_rotation_target"]
            delta_angle = calculate_min_rotation(current_angle, expected_body_x)
            control = "A" if delta_angle > 0 else "D"
            angle_diff_deg = abs(delta_angle)
            logging.info(f"ESCAPING: angle_diff={angle_diff_deg:.2f}°, control={control}, expected_body_x={expected_body_x:.2f}")
            if abs(delta_angle) < ROTATION_THRESHOLD_DEG:
                player_state["state"] = "PAUSE"
                player_state["pause_start_time"] = current_time
                player_state["escape_rotation_target"] = None
                player_state["escape_start_time"] = None
                global grid
                grid = Grid()
                logging.info(f"ESCAPING completed: body_x={player_state['body_x']:.2f}")
            elif player_state["escape_start_time"] and (current_time - player_state["escape_start_time"]) > ROTATION_TIMEOUT:
                player_state["state"] = "PAUSE"
                player_state["pause_start_time"] = current_time
                player_state["escape_rotation_target"] = None
                player_state["escape_start_time"] = None
                grid = Grid()
                logging.info(f"ESCAPING completed: body_x={player_state['body_x']:.2f} (timeout)")
            else:
                weight = calculate_rotation_weight(angle_diff_deg)

        elif player_state["state"] == "PAUSE":
            if (current_time - player_state["pause_start_time"]) >= PAUSE_DURATION:
                player_state["state"] = "MOVING"
                control = "W"
                weight = calculate_move_weight(player_state["distance_to_destination"])
                threading.Thread(target=lambda: asyncio.run(run_async_task()), daemon=True).start()
                logging.info(f"PAUSE -> MOVING, control={control}, weight={weight:.2f}")
            else:
                control = "STOP"
                weight = 0.0

        elif player_state["state"] == "MOVING":
            distance = distance(player_state["position"], player_state["destination"])
            if distance <= 1.0:
                player_state["state"] = "STOPPED"
                player_state["destination"] = None
                player_state["last_move_time"] = None
                player_state["last_move_position"] = None
                logging.info(f"Arrived at destination: {player_state['destination']}")
            else:
                target_angle = calculate_target_angle(player_state["position"], player_state["destination"])
                delta_angle = calculate_min_rotation(current_angle, target_angle)
                angle_diff_deg = abs(delta_angle)
                if angle_diff_deg > 45:
                    player_state["state"] = "ROTATING"
                    player_state["rotation_start_time"] = current_time
                    control = "A" if delta_angle > 0 else "D"
                    weight = calculate_rotation_weight(angle_diff_deg)
                    logging.info(f"MOVING -> ROTATING: angle_to_dest={target_angle:.2f}°, body_x={current_angle:.2f}, angle_diff={angle_diff_deg:.2f}°")
                else:
                    control = "W"
                    weight = calculate_move_weight(distance)
                    logging.info(f"MOVING: control={control}, weight={weight:.2f}, angle_diff={angle_diff_deg:.2f}°")

        elif player_state["state"] == "STOPPED":
            control = "STOP"
            weight = 0.0
            logging.info("STOPPED")

        player_state["last_position"] = player_state["position"]

    response = {"status": "success", "control": control, "weight": weight}
    if expected_body_x is not None:
        response["expected_body_x"] = expected_body_x
    return jsonify(response)

@app.route('/update_position', methods=['POST'])
def update_position():
    data = request.get_json()
    if not data or "position" not in data:
        logging.error("Missing position data")
        return jsonify({"status": "ERROR", "message": "Missing position data"}), 400
    try:
        x, y, z = map(float, data["position"].split(","))
        with state_lock:
            if player_state["last_position"] and distance((x, z), player_state["last_position"]) > 10.0:
                logging.error(f"Position jump detected: ({player_state['last_position']} -> ({x}, {z}))")
                return jsonify({"status": "ERROR", "message": "Position jump detected"}), 400
            if player_state["destination"]:
                dest_dist = distance((x, z), player_state["destination"])
                if dest_dist <= 1.0:
                    player_state["state"] = "STOPPED"
                    player_state["destination"] = None
                    player_state["last_move_time"] = None
                    player_state["last_move_position"] = None
                    logging.info(f"Arrived at destination via update_position: {player_state['destination']}")
            player_state["position"] = (x, z)
            if player_state["last_position"]:
                dx, dz = x - player_state["last_position"][0], z - player_state["last_position"][1]
                logging.info(f"Movement: dx={dx:.6f}, dz={dz:.6f}")
        logging.info(f"Position: {player_state['position']}")
        return jsonify({"status": "OK", "current_position": player_state["position"]})
    except Exception as e:
        logging.error(f"Update position failed: {e}")
        return jsonify({"status": "ERROR", "message": str(e)}), 400

@app.route('/get_move', methods=['GET'])
def get_move():
    with state_lock:
        if not player_state["waypoints"]:
            return jsonify({"move": "STOP", "weight": 0.0})
        
        next_point = pure_pursuit_target(player_state["position"], player_state["waypoints"])
        
        if next_point is None:
            player_state["state"] = "STOPPED"
            player_state["destination"] = None
            player_state["waypoints"] = []
            return jsonify({"move": "STOP", "weight": 0.0})

        player_state["current_target"] = next_point
      
        if player_state["state"] == "STOPPED":
            return jsonify({"move": "STOP", "weight": 0.0})
        
        elif player_state["state"] == "MOVING":
            distance = distance(player_state["position"], player_state["destination"])
            weight = calculate_move_weight(distance)
            
            if distance < 1.0:
                player_state["state"] = "STOPPED"
                return jsonify({"move": "STOP", "weight": 0.0})

            return jsonify({"move": "W", "weight": weight})
        
        elif player_state["state"] in ["ROTATING", "ESCAPING"]:
            current_angle = player_state["body_x"]
            target_angle = player_state.get("escape_rotation_target", calculate_target_angle(player_state["position"], player_state["destination"]))
            delta_angle = calculate_min_rotation(current_angle, target_angle)
            control = "A" if delta_angle > 0 else "D"
            weight = calculate_rotation_weight(abs(delta_angle))
            return jsonify({"move": control, "weight": weight})
        
        elif player_state["state"] == "PAUSE":
            return jsonify({"move": "STOP", "weight": 0.0})
        else:
            return jsonify({"move": "STOP", "weight": 0.0})

@app.route('/get_action', methods=['GET'])
def get_action():
    with state_lock:
        if player_state["enemy_detected"]:
            logging.info(f"FIRE: target={player_state['last_shot_target']}")
            player_state["enemy_detected"] = False
            return jsonify({"turret": "FIRE", "weight": 1.0})
        logging.info("No enemy detected")
        return jsonify({"turret": "", "weight": 0.0})

@app.route('/update_bullet', methods=['POST'])
def update_bullet():
    data = request.get_json()
    if not data:
        logging.error("Invalid bullet data")
        return jsonify({"status": "ERROR", "message": "Invalid request data"}), 400
    logging.info(f"Bullet: X={data.get('x')}, Y={data.get('y')}, Z={data.get('z')}, Target={data.get('hit')}")
    return jsonify({"status": "OK", "message": "Bullet impact data received"})

@app.route('/set_destination', methods=['POST'])
def set_destination():
    data = request.get_json()
    if not data or "destination" not in data:
        logging.error("Missing destination data")
        return jsonify({"status": "ERROR", "message": "Missing destination data"}), 400
    try:
        x, y, z = map(float, data["destination"].split(","))
        if not (0 <= x < grid.width and 0 <= z < grid.height):
            logging.error(f"Destination ({x}, {z}) out of bounds")
            return jsonify({"status": "ERROR", "message": f"Destination ({x}, {z}) out of bounds"}), 400
        with state_lock:
            player_state["destination"] = (x, z)
            player_state["waypoints"] = a_star(player_state["position"], (x, z), grid)
            player_state["state"] = "ROTATING"
            player_state["rotation_start_time"] = time.time()
            player_state["last_move_time"] = None
            player_state["last_move_position"] = None
        logging.info(f"Destination: ({x}, {z})")
        return jsonify({"status": "OK", "destination": {"x": x, "y": y, "z": z}})
    except Exception as e:
        logging.error(f"Set destination failed: {e}")
        return jsonify({"status": "ERROR", "message": str(e)}), 400

@app.route('/update_obstacle', methods=['POST'])
def update_obstacle():
    global obstacles, grid
    data = request.get_json()
    if not data:
        logging.error("No obstacle data received")
        return jsonify({'status': 'error', 'message': 'No data received'}), 400
    obstacles = data.get('obstacles', [])
    for obs in obstacles:
        if "image" not in obs:
            logging.warning(f"Obstacle missing image data: {obs}")
    grid = Grid()
    with state_lock:
        start_pos = player_state["position"]
        goal_pos = player_state["destination"]
    for obstacle in obstacles:
        detection = asyncio.run(analyze_obstacle(obstacle, 0))
        if detection["className"] in OBSTACLE_CLASSES and detection["confidence"] >= CONFIDENCE_THRESHOLD:
            grid.set_obstacle(obstacle["x_min"], obstacle["x_max"], obstacle["z_min"], obstacle["z_max"], start_pos, goal_pos)
    with state_lock:
        if player_state["destination"]:
            player_state["waypoints"] = a_star(player_state["position"], player_state["destination"], grid)
    logging.info(f"Obstacles: {len(obstacles)}")
    return jsonify({'status': 'success', 'message': 'Obstacle data received'})

@app.route('/init', methods=['GET'])
def init():
    with state_lock:
        player_state["state"] = "IDLE"
        player_state["destination"] = None
        player_state["position"] = (60.0, 27.23)
        player_state["body_x"] = 0.0
        player_state["last_position"] = None
        player_state["rotation_start_time"] = None
        player_state["pause_start_time"] = None
        player_state["last_body_x"] = None
        player_state["enemy_detected"] = False
        player_state["last_move_time"] = None
        player_state["last_move_position"] = None
        player_state["escape_rotation_target"] = None
        player_state["escape_start_time"] = None
        player_state["shots_fired"] = 0
    config = {
        "startMode": "start",
        "blStartX": 60,
        "blStartY": 10,
        "blStartZ": 27.23,
        "rdStartX": 59,
        "rdStartY": 10,
        "rdStartZ": 280
    }
    logging.info("Initialized")
    return jsonify(config)

@app.route('/start', methods=['GET'])
def start():
    logging.info("Start")
    return jsonify({"control": ""})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)