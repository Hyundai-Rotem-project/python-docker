from flask import Flask, request, jsonify
import math
import torch
from ultralytics import YOLO
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
import matplotlib.pyplot as plt
import matplotlib.patches as patches

app = Flask(__name__)

# 로깅 설정
logging.basicConfig(filename='tank.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# YOLO 모델 로드
try:
    model = YOLO('best.pt')
except Exception as e:
    logging.error(f"YOLO model load failed: {e}")
    raise

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

    def node_from_world_point(self, world_x, world_z):
        grid_x = max(0, min(int(round(world_x)), self.width - 1))
        grid_z = max(0, min(int(round(world_z)), self.height - 1))
        return self.grid[grid_x][grid_z]

    def set_obstacle(self, x_min, x_max, z_min, z_max, start_pos=None, goal_pos=None):
        x_min = max(0, min(int(round(x_min)), self.width - 1))
        x_max = max(0, min(int(round(x_max)), self.width - 1))
        z_min = max(0, min(int(round(z_min)), self.height - 1))
        z_max = max(0, min(int(round(z_max)), self.height - 1))
        for x in range(x_min, x_max + 1):
            for z in range(z_min, z_max + 1):
                if start_pos and goal_pos:
                    if (x, z) == (int(round(start_pos[0])), int(round(start_pos[1]))) or \
                       (x, z) == (int(round(goal_pos[0])), int(round(goal_pos[1]))):
                        continue
                self.grid[x][z].is_obstacle = True
        print(f"🪨 Obstacle set: x_min={x_min}, x_max={x_max}, z_min={z_min}, z_max={z_max}")
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
    "shot_cooldown": 2.0,
    "body_x": 0.0,
    "body_y": 0.0,
    "body_z": 0.0,
    "last_valid_angle": None,
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

# 상수
ROTATION_THRESHOLD_DEG = 10
ROTATION_TIMEOUT = 2.0
PAUSE_DURATION = 0.5
WEIGHT_LEVELS = [1.0, 0.6, 0.3, 0.1, 0.05, 0.01]
DETECTION_RANGE = 100.0
FIRING_RANGE = 45.0  # 포격 가능 거리
ENEMY_CLASSES = {'car2', 'car3', 'tank'}
FRIENDLY_CLASSES = {'car5'}
OBSTACLE_CLASSES = {'rock1', 'rock2', 'wall1', 'wall2','human1'}
CONFIDENCE_THRESHOLD = 0.5  # 신뢰도 임계값 낮춤 (디버깅 용)
TRAPPED_TIMEOUT = 1.0
ESCAPE_ROTATION_ANGLE = 90.0
SHOTS_PER_ENEMY = 1  
SHOTS_PER_OBSTACLE = 0
SHOT_INTERVAL = 1.0

def select_weight(value, levels=WEIGHT_LEVELS):
    return min(levels, key=lambda x: abs(x - value))

def calculate_move_weight(distance):
    return select_weight(1.0)

def calculate_rotation_weight(angle_diff_deg):
    abs_deg = abs(angle_diff_deg)
    if abs_deg < ROTATION_THRESHOLD_DEG:
        return 0.0
    target_weight = min(0.3, abs_deg / 90)
    return select_weight(target_weight)

def calculate_min_rotation(current_angle, target_angle):
    """최소 회전 각도 계산 (도 단위)"""
    delta_angle = (target_angle - current_angle + 180) % 360 - 180
    # delta_angle = (target_angle - current_angle + np.pi) % (2 * np.pi) - np.pi
    return delta_angle

def calculate_target_angle(current_pos, target_pos):
    """목표 방향 각도 계산 (도 단위)"""
    dx = target_pos[0] - current_pos[0]
    dz = target_pos[1] - current_pos[1]
    return math.degrees(math.atan2(dz, dx)) % 360

async def analyze_obstacle(obstacle, index):
    x_center = (obstacle["x_min"] + obstacle["x_max"]) / 2
    z_center = (obstacle["z_min"] + obstacle["z_max"]) / 2
    image_data = obstacle.get("image")
    target_classes = {0: 'car2', 1: 'car3', 2: 'car5', 3: 'human1', 4: 'rock1', 5: 'rock2', 6: 'tank', 7: 'wall1', 8: 'wall2'}
    class_name = 'unknown'
    confidence = 0.0

    if not image_data:
        print(f"🔍 YOLO: No image, classified as unknown at ({x_center:.2f}, {z_center:.2f})")
        logging.info(f"YOLO: No image, classified as unknown at ({x_center:.2f}, {z_center:.2f})")
        return {"className": class_name, "position": (x_center, z_center), "confidence": confidence}

    try:
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(f"debug_image_{index}_{int(time.time())}.png")
        print(f"📸 Saved debug image: debug_image_{index}_{int(time.time())}.png")
        logging.info(f"Saved debug image: debug_image_{index}_{int(time.time())}.png")
        results = model.predict(image, verbose=False, conf=CONFIDENCE_THRESHOLD)
        detections = results[0].boxes.data.cpu().numpy()
        print(f"🔍 YOLO raw detections: {detections}")
        logging.info(f"YOLO raw detections: {detections}")
        filtered_results = [
            {
                'className': target_classes[int(box[5])],
                'confidence': float(box[4]),
                'bbox': [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
            }
            for box in detections if int(box[5]) in target_classes
        ]
        if filtered_results:
            detection = max(filtered_results, key=lambda x: x['confidence'])
            class_name = detection['className']
            confidence = detection['confidence']
            print(f"🔍 YOLO: {class_name} at ({x_center:.2f}, {z_center:.2f}), confidence={confidence:.2f}")
            logging.info(f"YOLO: {class_name} at ({x_center:.2f}, {z_center:.2f}), confidence={confidence:.2f}")
        else:
            print(f"🔍 YOLO: No valid detections at ({x_center:.2f}, {z_center:.2f})")
            logging.info(f"YOLO: No valid detections at ({x_center:.2f}, {z_center:.2f})")
    except Exception as e:
        print(f"🔍 YOLO failed: {e}")
        logging.error(f"YOLO failed: {e}")
        class_name = 'unknown'

    if class_name in ENEMY_CLASSES and confidence >= CONFIDENCE_THRESHOLD:
        print(f"🔫 Enemy detected: {class_name} at ({x_center:.2f}, {z_center:.2f}), confidence={confidence:.2f}")
        logging.info(f"Enemy detected: {class_name} at ({x_center:.2f}, {z_center:.2f}), confidence={confidence:.2f}")
    elif class_name in FRIENDLY_CLASSES:
        print(f"🤝 Friendly detected: {class_name} at ({x_center:.2f}, {z_center:.2f}), confidence={confidence:.2f}")
        logging.info(f"Friendly detected: {class_name} at ({x_center:.2f}, {z_center:.2f}), confidence={confidence:.2f}")
    elif class_name in OBSTACLE_CLASSES:
        print(f"🪨 Obstacle detected: {class_name} at ({x_center:.2f}, {z_center:.2f}), confidence={confidence:.2f}")
        logging.info(f"Obstacle detected: {class_name} at ({x_center:.2f}, {z_center:.2f}), confidence={confidence:.2f}")
    else:
        print(f"❓ Unknown object at ({x_center:.2f}, {z_center:.2f}), class={class_name}")
        logging.info(f"Unknown object at ({x_center:.2f}, {z_center:.2f}), class={class_name}")

    return {"className": class_name, "position": (x_center, z_center), "confidence": confidence}

async def shoot_at_target(target_pos, target_type="enemy"):
    with state_lock:
        current_time = asyncio.get_event_loop().time()
        player_state["last_shot_time"] = current_time - player_state["shot_cooldown"]
        player_state["last_shot_target"] = target_pos
        bullet_data = {"x": target_pos[0], "y": 0.0, "z": target_pos[1], "hit": target_type}
        for attempt in range(2):
            try:
                response = requests.post('http://localhost:5000/update_bullet', json=bullet_data, timeout=5)
                if response.status_code == 200:
                    player_state["shots_fired"] += 1
                    print(f"🔫 Shot #{player_state['shots_fired']} fired at {target_type} {target_pos}")
                    logging.info(f"Shot #{player_state['shots_fired']} fired at {target_type} {target_pos}")
                    return bullet_data
                else:
                    print(f"🔫 Shot failed: HTTP {response.status_code}")
                    logging.error(f"Shot failed: HTTP {response.status_code}")
            except requests.RequestException as e:
                print(f"🔫 Shot failed, attempt {attempt+1}: {e}")
                logging.error(f"Shot failed, attempt {attempt+1}: {e}")
        print(f"🔫 Shot failed: All attempts failed")
        logging.error(f"Shot failed: All attempts failed")
        return None

def heuristic_point(node, goal_node):
    """
    p1: (x1, z1)
    p2: (x2, z2)
    => 2D 거리 계산 (y축 무시)
    """
    return math.sqrt((node.x - goal_node.x)**2 + (node.z - goal_node.z)**2)

def a_star(start, goal, grid):
    def heuristic(node, goal_node):
        return math.sqrt((node.x - goal_node.x) ** 2 + (node.z - goal_node.z) ** 2)

    if not (0 <= start[0] < grid.width and 0 <= start[1] < grid.height):
        print(f"🚫 A* failed: Invalid start {start}")
        logging.error(f"A* failed: Invalid start {start}")
        return []
    if not (0 <= goal[0] < grid.width and 0 <= goal[1] < grid.height):
        print(f"🚫 A* failed: Invalid goal {goal}")
        logging.error(f"A* failed: Invalid goal {goal}")
        return []

    start_node = grid.node_from_world_point(start[0], start[1])
    goal_node = grid.node_from_world_point(goal[0], goal[1])

    if start_node.is_obstacle:
        print(f"🚫 A* failed: Start is obstacle at ({start_node.x}, {start_node.z})")
        logging.error(f"A* failed: Start is obstacle at ({start_node.x}, {start_node.z})")
        return []
    if goal_node.is_obstacle:
        print(f"🚫 A* failed: Goal is obstacle at ({goal_node.x}, {goal_node.z})")
        logging.error(f"A* failed: Goal is obstacle at ({goal_node.x}, {goal_node.z})")
        return []

    open_set = []
    counter = 0
    heapq.heappush(open_set, (0 + heuristic(start_node, goal_node), 0, counter, start_node))
    came_from = {}
    g_score = {start_node: 0}
    f_score = {start_node: heuristic(start_node, goal_node)}

    while open_set:
        _, current_g, _, current = heapq.heappop(open_set)
    print(f"🚫 A* failed: No path from ({start_node.x}, {start_node.z}) to ({goal_node.x}, {goal_node.z})")
    logging.error(f"A* failed: No path from ({start_node.x}, {start_node.z}) to ({goal_node.x}, {goal_node.z})")
    return []

def pure_pursuit_target(current_pos, path, lookahead_distance=5.0):
    """
    현재 위치 기준 path에서 lookahead_distance 이상 떨어진 지점 반환.
    너무 가까우면 path pop하고 다음 지점 추적.

    Args:
        current_pos (tuple): (x, z) 현재 위치
        path (list of tuple): [(x1, z1), (x2, z2), ...]
        lookahead_distance (float): 최소 거리
    Returns:
        (float, float): 추적할 타겟 좌표
    """
    if not path:
        return current_pos

    x0, z0 = current_pos

    while len(path) > 1:
        px, pz = path[0]
        distance = math.hypot(px - x0, pz - z0)
        if distance < lookahead_distance:
            # 현재 포인트 너무 가까우면 버리고 다음 포인트 추적
            path.pop(0)
        else:
            break

    # 현재 남아있는 첫 번째 포인트를 목표로 삼는다
    return path[0]


def move_to_firing_range(current_pos, enemy_pos, firing_range=FIRING_RANGE, grid=grid):
    """포격 가능 거리 내로 이동 (A* 경로 사용)"""
    distance = math.hypot(enemy_pos[0] - current_pos[0], enemy_pos[1] - current_pos[1])
    if distance <= firing_range:
        logging.info(f"이미 포격 가능 거리 내: {distance:.2f}m")
        return current_pos
    else:
        # 포격 가능 거리 내의 목표 지점 계산
        direction = [(enemy_pos[0] - current_pos[0]) / distance, (enemy_pos[1] - current_pos[1]) / distance]
        target_pos = [
            current_pos[0] + direction[0] * (distance - firing_range),
            current_pos[1] + direction[1] * (distance - firing_range)
        ]
        path = a_star(current_pos, target_pos, grid)
        if not path:
            logging.warning(f"No path to firing range: {target_pos}")
            return current_pos
        next_pos = path[1] if len(path) > 1 else target_pos
        logging.info(f"포격 가능 거리({firing_range}m)로 이동: {next_pos}")
        return next_pos

async def move_towards_destination():
    global obstacles, grid
    print("🚀 Starting move_towards_destination")
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
        distance = math.hypot(dest[0] - current_pos[0], dest[1] - current_pos[1])
        print(f"🚗 Pos={current_pos}, Dest={dest}, Distance={distance:.2f}, State={state}, BodyX={current_angle:.2f}")
        logging.info(f"Pos={current_pos}, Dest={dest}, Distance={distance:.2f}, State={state}, BodyX={current_angle:.2f}")

        if distance <= 1.0:
            with state_lock:
                player_state["state"] = "STOPPED"
                player_state["destination"] = None
                player_state["last_move_time"] = None
                player_state["last_move_position"] = None
            print(f"🎉 Arrived at destination: {dest}")
            logging.info(f"Arrived at destination: {dest}")
            break

        if state not in ["MOVING", "ESCAPING", "PAUSE", "ROTATING"]:
            await asyncio.sleep(0.2)
            continue

        # 갇힘 감지
        if state == "MOVING":
            move_distance = math.hypot(current_pos[0] - player_state["last_move_position"][0],
                                      current_pos[1] - player_state["last_move_position"][1])
            if current_time - player_state["last_move_time"] > TRAPPED_TIMEOUT and move_distance < 0.5:
                with state_lock:
                    player_state["state"] = "ESCAPING"
                    player_state["escape_rotation_target"] = (current_angle + ESCAPE_ROTATION_ANGLE) % 360
                    player_state["escape_start_time"] = current_time
                    player_state["last_move_time"] = current_time
                    player_state["last_move_position"] = current_pos
                print(f"🚫 Trapped detected: No movement for {TRAPPED_TIMEOUT}s at {current_pos}")
                print(f"🔄 Starting ESCAPING: Rotating to {player_state['escape_rotation_target']:.2f}°")
                logging.info(f"Trapped detected: No movement for {TRAPPED_TIMEOUT}s at {current_pos}")
                logging.info(f"Starting ESCAPING: Rotating to {player_state['escape_rotation_target']:.2f}°")
                await asyncio.sleep(0.2)
                continue

        # 적 처리
        target_detected = False
        print(f"🔍 Checking {len(obstacles)} obstacles")
        logging.info(f"Checking {len(obstacles)} obstacles")
        for idx, obstacle in enumerate(obstacles):
            obs_center = ((obstacle["x_min"] + obstacle["x_max"]) / 2, (obstacle["z_min"] + obstacle["z_max"]) / 2)
            obs_distance = math.hypot(obs_center[0] - current_pos[0], obs_center[1] - current_pos[1])
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
                    # 적 방향으로 회전
                    target_angle = calculate_target_angle(current_pos, obs_center)
                    delta_angle = calculate_min_rotation(current_angle, target_angle)
                    control = "A" if delta_angle > 0 else "D"
                    weight = calculate_rotation_weight(abs(delta_angle))
                    print(f"🔫 Enemy detected at {obs_center}, distance={obs_distance:.2f}m, rotating to {target_angle:.2f}°")
                    logging.info(f"Enemy detected at {obs_center}, distance={obs_distance:.2f}m, rotating to {target_angle:.2f}°")
                    if abs(delta_angle) < ROTATION_THRESHOLD_DEG:
                        # 회전 완료, 포격 가능 거리로 이동
                        new_pos = move_to_firing_range(current_pos, obs_center)
                        with state_lock:
                            player_state["position"] = new_pos
                            player_state["last_move_time"] = current_time
                            player_state["last_move_position"] = new_pos
                        # 포격
                        await shoot_at_target(obs_center, target_type="enemy")
                        print(f"🔫 Fired at enemy at {obs_center}, moving to destination")
                        logging.info(f"Fired at enemy at {obs_center}, moving to destination")
                        # Grid 갱신 및 경로 재탐색
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
                        # 목적지 방향으로 회전
                        target_angle = calculate_target_angle(new_pos, dest)
                        delta_angle = calculate_min_rotation(current_angle, target_angle)
                        control = "A" if delta_angle > 0 else "D"
                        weight = calculate_rotation_weight(abs(delta_angle))
                        if abs(delta_angle) < ROTATION_THRESHOLD_DEG:
                            player_state["state"] = "MOVING"
                            control = "W"
                            weight = calculate_move_weight(distance)
                        break
                elif detection["className"] in OBSTACLE_CLASSES and detection["confidence"] >= CONFIDENCE_THRESHOLD:
                    continue
                else:
                    print(f"🔫 Skipped shooting: Not an enemy or low confidence ({detection['className']}, confidence={detection['confidence']:.2f})")
                    logging.info(f"Skipped shooting: Not an enemy or low confidence ({detection['className']}, confidence={detection['confidence']:.2f})")

        if target_detected:
            await asyncio.sleep(PAUSE_DURATION)
            continue

        # 경로 계산
        path = a_star(current_pos, dest, grid)
        if not path:
            print(f"🚫 A* failed, attempt {attempt+1}/{max_attempts}")
            logging.error(f"A* failed, attempt {attempt+1}/{max_attempts}")
            attempt += 1
            if attempt >= max_attempts:
                dx, dz = dest[0] - current_pos[0], dest[1] - current_pos[1]
                dist = math.hypot(dx, dz)
                if dist > 1.0 and dist > 1e-6:
                    move_dist = min(1.0, dist)
                    next_x = current_pos[0] + (dx / dist) * move_dist
                    next_z = current_pos[1] + (dz / dist) * move_dist
                    if not check_obstacle_collision(next_x, next_z, obstacles):
                        with state_lock:
                            player_state["position"] = (next_x, next_z)
                            player_state["last_move_time"] = current_time
                            player_state["last_move_position"] = (next_x, next_z)
                        print(f"🚗 Fallback: Moved to ({next_x:.2f}, {next_z:.2f})")
                        logging.info(f"Fallback: Moved to ({next_x:.2f}, {next_z:.2f})")
                    else:
                        print(f"🚫 Fallback: Collision at ({next_x:.2f}, {next_z:.2f})")
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
                    print("🚫 Max attempts reached or too close")
                    logging.error("Max attempts reached or too close")
                    break
                await asyncio.sleep(1.0)
            continue

        attempt = 0
        next_pos = path[1] if len(path) > 1 else dest
        dx = next_pos[0] - current_pos[0]
        dz = next_pos[1] - current_pos[1]
        dist = math.hypot(dx, dz)
        if dist > 1e-6:
            angle_to_next = math.degrees(math.atan2(dz, dx)) % 360
            delta_angle = calculate_min_rotation(current_angle, angle_to_next)
            if abs(delta_angle) > 135:
                print(f"🚫 Skipping waypoint {next_pos}: delta_angle={delta_angle:.2f}°, too far from body_x={current_angle:.2f}")
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
        print(f"🚗 Moved to waypoint: {next_pos}, angle_to_next={angle_to_next:.2f}°")
        logging.info(f"Moved to waypoint: {next_pos}, angle_to_next={angle_to_next:.2f}°")
        await asyncio.sleep(0.2)
    print("🏁 Movement stopped")
    logging.info("Movement stopped")

def check_obstacle_collision(x, z, obstacles):
    for obs in obstacles:
        if obs["x_min"] <= x <= obs["x_max"] and obs["z_min"] <= z <= obs["z_max"]:
            print(f"🚫 Collision at ({x:.2f}, {z:.2f})")
            logging.error(f"Collision at ({x:.2f}, {z:.2f})")
            return True
    return False

def run_async_task():
    retries = 3
    for attempt in range(retries):
        try:
            _template = asyncio.new_event_loop()
            asyncio.set_event_loop(_template)
            _template.run_until_complete(move_towards_destination())
            break
        except Exception as e:
            print(f"🚫 Async task failed (attempt {attempt+1}/{retries}): {e}, destination={player_state['destination']}")
            logging.error(f"Async task failed (attempt {attempt+1}/{retries}): {e}, destination={player_state['destination']}")
            if attempt < retries - 1:
                time.sleep(1)
        finally:
            _template.close()

@app.route('/detect', methods=['POST'])
def detect():
    image = request.files.get('image')
    if not image:
        print("🔍 YOLO: No image")
        logging.error("YOLO: No image")
        return jsonify({"error": "No image received"}), 400
    try:
        image = Image.open(image)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        results = model.predict(image, verbose=False, conf=CONFIDENCE_THRESHOLD)
        detections = results[0].boxes.data.cpu().numpy()
        target_classes = {0: 'car2', 1: 'car3', 2: 'car5', 3: 'human1', 4: 'rock1', 5: 'rock2', 6: 'tank', 7: 'wall1', 8: 'wall2'}
        filtered_results = [
            {
                'className': target_classes[int(box[5])],
                'bbox': [float(coord) for coord in box[:4]],
                'confidence': float(box[4]),
                'x_min': float(box[0]),
                'x_max': float(box[2]),
                'z_min': float(box[1]),
                'z_max': float(box[3]),
                'image': image_base64
            }
            for box in detections if int(box[5]) in target_classes
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
        print(f"🔍 YOLO: {len(filtered_results)} detections, updated obstacles")
        logging.info(f"YOLO: {len(filtered_results)} detections, updated obstacles")
        return jsonify(filtered_results)
    except Exception as e:
        print(f"🔍 YOLO failed: {e}")
        logging.error(f"YOLO failed: {e}")
        return jsonify({"error": "Detection failed"}), 500

@app.route('/info', methods=['POST'])
def info():
    data = request.get_json(force=True)
    if not data:
        print("🚫 No JSON received")
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
                    print(f"⚠️ Forced body_x update to {player_state['body_x']:.2f} after delay, expected={target:.2f}")
                    logging.warning(f"Forced body_x update to {player_state['body_x']:.2f} after delay, expected={target:.2f}")
                print(f"⚠️ bodyX change too small during {player_state['state']}: ΔbodyX={dbx:.3f}, body_x={body_x:.2f}")
                logging.warning(f"bodyX change too small during {player_state['state']}: ΔbodyX={dbx:.3f}, body_x={body_x:.2f}")
            print(f"🔄 ΔbodyX={dbx:.3f}")
            logging.info(f"ΔbodyX={dbx:.3f}")
        player_state["last_body_x"] = body_x

    if not player_state["destination"]:
        with state_lock:
            player_state["state"] = "IDLE"
            player_state["last_move_time"] = None
            player_state["last_move_position"] = None
        print("🛑 IDLE: No destination")
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
            print("🔄 IDLE -> ROTATING")
            logging.info("IDLE -> ROTATING")

        elif player_state["state"] == "ROTATING":
            dx, dz = player_state["destination"]
            px, pz = player_state["position"]
            target_angle = calculate_target_angle(player_state["position"], player_state["destination"])
            delta_angle = calculate_min_rotation(current_angle, target_angle)
            expected_body_x = target_angle
            control = "A" if delta_angle > 0 else "D"
            angle_diff_deg = abs(delta_angle)
            print(f"🧭 ROTATING: angle_diff={angle_diff_deg:.2f}°, control={control}, expected_body_x={expected_body_x:.2f}")
            logging.info(f"ROTATING: angle_diff={angle_diff_deg:.2f}°, control={control}, expected_body_x={expected_body_x:.2f}")
            if abs(delta_angle) < ROTATION_THRESHOLD_DEG:
                player_state["state"] = "PAUSE"
                player_state["pause_start_time"] = current_time
                print(f"🔄 ROTATING -> PAUSE: body_x={player_state['body_x']:.2f}, expected_body_x={expected_body_x:.2f}")
                logging.info(f"ROTATING -> PAUSE: body_x={player_state['body_x']:.2f}, expected_body_x={expected_body_x:.2f}")
            elif player_state["rotation_start_time"] and (current_time - player_state["rotation_start_time"]) > ROTATION_TIMEOUT:
                player_state["state"] = "PAUSE"
                player_state["pause_start_time"] = current_time
                print("🔄 ROTATING -> PAUSE: timeout")
                logging.info("ROTATING -> PAUSE: timeout")
            else:
                weight = calculate_rotation_weight(angle_diff_deg)

        elif player_state["state"] == "ESCAPING":
            expected_body_x = player_state["escape_rotation_target"]
            delta_angle = calculate_min_rotation(current_angle, expected_body_x)
            control = "A" if delta_angle > 0 else "D"
            angle_diff_deg = abs(delta_angle)
            print(f"🔄 ESCAPING: angle_diff={angle_diff_deg:.2f}°, control={control}, expected_body_x={expected_body_x:.2f}")
            logging.info(f"ESCAPING: angle_diff={angle_diff_deg:.2f}°, control={control}, expected_body_x={expected_body_x:.2f}")
            if abs(delta_angle) < ROTATION_THRESHOLD_DEG:
                player_state["state"] = "PAUSE"
                player_state["pause_start_time"] = current_time
                player_state["escape_rotation_target"] = None
                player_state["escape_start_time"] = None
                global grid
                grid = Grid()
                print(f"✅ ESCAPING completed: body_x={player_state['body_x']:.2f}")
                logging.info(f"ESCAPING completed: body_x={player_state['body_x']:.2f}")
            elif player_state["escape_start_time"] and (current_time - player_state["escape_start_time"]) > ROTATION_TIMEOUT:
                player_state["state"] = "PAUSE"
                player_state["pause_start_time"] = current_time
                player_state["escape_rotation_target"] = None
                player_state["escape_start_time"] = None
                grid = Grid()
                print(f"✅ ESCAPING completed: body_x={player_state['body_x']:.2f} (timeout)")
                logging.info(f"ESCAPING completed: body_x={player_state['body_x']:.2f} (timeout)")
            else:
                weight = calculate_rotation_weight(angle_diff_deg)

        elif player_state["state"] == "PAUSE":
            if (current_time - player_state["pause_start_time"]) >= PAUSE_DURATION:
                player_state["state"] = "MOVING"
                control = "W"
                weight = calculate_move_weight(player_state["distance_to_destination"])
                threading.Thread(target=run_async_task, daemon=True).start()
                print(f"🔄 PAUSE -> MOVING, control={control}, weight={weight:.2f}")
                logging.info(f"PAUSE -> MOVING, control={control}, weight={weight:.2f}")
            else:
                control = "STOP"
                weight = 0.0

        elif player_state["state"] == "MOVING":
            dx, dz = player_state["destination"]
            px, pz = player_state["position"]
            distance = math.hypot(dx - px, dz - pz)
            if distance <= 1.0:
                player_state["state"] = "STOPPED"
                player_state["destination"] = None
                player_state["last_move_time"] = None
                player_state["last_move_position"] = None
                print(f"🎉 Arrived at destination: {player_state['destination']}")
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
                    print(f"🔄 MOVING -> ROTATING: angle_to_dest={target_angle:.2f}°, body_x={current_angle:.2f}, angle_diff={angle_diff_deg:.2f}°")
                    logging.info(f"MOVING -> ROTATING: angle_to_dest={target_angle:.2f}°, body_x={current_angle:.2f}, angle_diff={angle_diff_deg:.2f}°")
                else:
                    control = "W"
                    weight = calculate_move_weight(distance)
                    print(f"🚗 MOVING: control={control}, weight={weight:.2f}, angle_diff={angle_diff_deg:.2f}°")
                    logging.info(f"MOVING: control={control}, weight={weight:.2f}, angle_diff={angle_diff_deg:.2f}°")

        elif player_state["state"] == "STOPPED":
            control = "STOP"
            weight = 0.0
            print("🛑 STOPPED")
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
        print("🚫 Missing position data")
        logging.error("Missing position data")
        return jsonify({"status": "ERROR", "message": "Missing position data"}), 400
    try:
        x, y, z = map(float, data["position"].split(","))
        with state_lock:
            if player_state["last_position"] and math.hypot(x - player_state["last_position"][0], z - player_state["last_position"][1]) > 10.0:
                print(f"⚠️ Position jump detected: ({player_state['last_position']} -> ({x}, {z}))")
                logging.error(f"Position jump detected: ({player_state['last_position']} -> ({x}, {z}))")
                return jsonify({"status": "ERROR", "message": "Position jump detected"}), 400
            if player_state["destination"]:
                dest_dist = math.hypot(x - player_state["destination"][0], z - player_state["destination"][1])
                if dest_dist <= 1.0:
                    player_state["state"] = "STOPPED"
                    player_state["destination"] = None
                    player_state["last_move_time"] = None
                    player_state["last_move_position"] = None
                    print(f"🎉 Arrived at destination via update_position: {player_state['destination']}")
                    logging.info(f"Arrived at destination via update_position: {player_state['destination']}")
            player_state["position"] = (x, z)
            if player_state["last_position"]:
                dx, dz = x - player_state["last_position"][0], z - player_state["last_position"][1]
                print(f"📍 Movement: dx={dx:.6f}, dz={dz:.6f}")
                logging.info(f"Movement: dx={dx:.6f}, dz={dz:.6f}")
        print(f"📍 Position: {player_state['position']}")
        logging.info(f"Position: {player_state['position']}")
        return jsonify({"status": "OK", "current_position": player_state["position"]})
    except Exception as e:
        print(f"🚫 Update position failed: {e}")
        logging.error(f"Update position failed: {e}")
        return jsonify({"status": "ERROR", "message": str(e)}), 400

@app.route('/get_move', methods=['GET'])
def get_move():
    with state_lock:
        if not player_state["waypoints"]:
            return jsonify({"move": "STOP", "weight": 0.0})
        
        # Pure pursuit target 계산
        next_point = pure_pursuit_target(player_state["position"], player_state["waypoints"])
        
        if next_point is None:
            # 경로를 다 따라 갔으면 STOP
            player_state["state"] = "STOPPED"
            player_state["destination"] = None
            player_state["waypoints"] = []
            return jsonify({"move": "STOP", "weight": 0.0})

         # 💥 여기 중요: next_point를 현재 목표처럼 사용
        target_x, target_z = next_point
        player_state["current_target"] = (target_x, target_z)
      
        # 현재 상태에 따라 다른 동작
        if player_state["state"] == "STOPPED":
            return jsonify({"move": "STOP", "weight": 0.0})
        
        elif player_state["state"] == "MOVING":
            #이동명령 생성
            distance = heuristic_point(player_state["position"], player_state["destination"])
            weight = calculate_move_weight(distance)
            # weight = calculate_move_weight(player_state["distance_to_destination"])
            
            # 도착하면 STOP
            if distance < 1.0:  # 목표 근처 도달 (예: 1m 이내)
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
            print(f"🔫 FIRE: target={player_state['last_shot_target']}")
            logging.info(f"FIRE: target={player_state['last_shot_target']}")
            player_state["enemy_detected"] = False
            return jsonify({"turret": "FIRE", "weight": 1.0})
        print("🔫 No enemy detected")
        logging.info("No enemy detected")
        return jsonify({"turret": "", "weight": 0.0})

@app.route('/update_bullet', methods=['POST'])
def update_bullet():
    data = request.get_json()
    if not data:
        print("🚫 Invalid bullet data")
        logging.error("Invalid bullet data")
        return jsonify({"status": "ERROR", "message": "Invalid request data"}), 400
    print(f"💥 Bullet: X={data.get('x')}, Y={data.get('y')}, Z={data.get('z')}, Target={data.get('hit')}")
    logging.info(f"Bullet: X={data.get('x')}, Y={data.get('y')}, Z={data.get('z')}, Target={data.get('hit')}")
    return jsonify({"status": "OK", "message": "Bullet impact data received"})

@app.route('/set_destination', methods=['POST'])
def set_destination():
    data = request.get_json()
    if not data or "destination" not in data:
        print("🚫 Missing destination data")
        logging.error("Missing destination data")
        return jsonify({"status": "ERROR", "message": "Missing destination data"}), 400
    try:
        x, y, z = map(float, data["destination"].split(","))
        if not (0 <= x < grid.width and 0 <= z < grid.height):
            print(f"🚫 Destination ({x}, {z}) out of bounds")
            logging.error(f"Destination ({x}, {z}) out of bounds")
            return jsonify({"status": "ERROR", "message": f"Destination ({x}, {z}) out of bounds"}), 400
        with state_lock:
            player_state["destination"] = (x, z)
            player_state["waypoints"] = a_star(player_state["position"], (x, z), grid)
            player_state["state"] = "ROTATING"
            player_state["rotation_start_time"] = time.time()
            player_state["last_valid_angle"] = None
            player_state["last_move_time"] = None
            player_state["last_move_position"] = None
        print(f"🎯 Destination: ({x}, {z})")
        logging.info(f"Destination: ({x}, {z})")
        return jsonify({"status": "OK", "destination": {"x": x, "y": y, "z": z}})
    except Exception as e:
        print(f"🚫 Set destination failed: {e}")
        logging.error(f"Set destination failed: {e}")
        return jsonify({"status": "ERROR", "message": str(e)}), 400

@app.route('/update_obstacle', methods=['POST'])
def update_obstacle():
    global obstacles, grid
    data = request.get_json()
    if not data:
        print("🚫 No obstacle data received")
        logging.error("No obstacle data received")
        return jsonify({'status': 'error', 'message': 'No data received'}), 400
    obstacles = data.get('obstacles', [])
    for obs in obstacles:
        if "image" not in obs:
            print(f"🚫 Obstacle missing image data: {obs}")
            logging.warning(f"Obstacle missing image data: {obs}")
    grid = Grid()
    with state_lock:
        start_pos = player_state["position"]
        goal_pos = player_state["destination"]
    for obstacle in obstacles:
        detection = asyncio.run(analyze_obstacle(obstacle, 0))
        if detection["className"] in OBSTACLE_CLASSES:
            grid.set_obstacle(obstacle["x_min"], obstacle["x_max"], obstacle["z_min"], obstacle["z_max"])
    # A* 경로 재탐색
    with state_lock:
        if player_state["destination"]:
            player_state["waypoints"] = a_star(player_state["position"], player_state["destination"], grid) 
    
    print(f"🪨 Obstacles: {len(obstacles)}")
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
        player_state["last_valid_angle"] = None
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
    print("🛠️ Initialized")
    logging.info("Initialized")
    return jsonify(config)

@app.route('/start', methods=['GET'])
def start():
    print("🚀 Start")
    logging.info("Start")
    return jsonify({"control": ""})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)