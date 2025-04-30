#1m 버퍼
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
import cv2

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
        # 1미터 버퍼 추가
        x_min = max(0, min(int(round(x_min - 1.0)), self.width - 1))
        x_max = max(0, min(int(round(x_max + 1.0)), self.width - 1))
        z_min = max(0, min(int(round(z_min - 1.0)), self.height - 1))
        z_max = max(0, min(int(round(z_max + 1.0)), self.height - 1))
        for x in range(x_min, x_max + 1):
            for z in range(z_min, z_max + 1):
                if start_pos and goal_pos:
                    if (x, z) == (int(round(start_pos[0])), int(round(start_pos[1]))) or \
                       (x, z) == (int(round(goal_pos[0])), int(round(goal_pos[1]))):
                        continue
                self.grid[x][z].is_obstacle = True
        print(f"🪨 Obstacle set with 1m buffer: x_min={x_min}, x_max={x_max}, z_min={z_min}, z_max={z_max}")
        logging.info(f"Obstacle set with 1m buffer: x_min={x_min}, x_max={x_max}, z_min={z_min}, z_max={z_max}")

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
PAUSE_DURATION = 3.0    
WEIGHT_LEVELS = [1.0, 0.6, 0.3, 0.1, 0.05, 0.01]
DETECTION_RANGE = 100.0
ENEMY_CLASSES = {'car002', 'car003', 'tank', 'enemy'}
FRIENDLY_CLASSES = {'car005'}
OBSTACLE_CLASSES = {'rock1', 'rock2', 'wall1', 'wall2', 'human1'}
CONFIDENCE_THRESHOLD = 0.5
TRAPPED_TIMEOUT = 1.0
ESCAPE_ROTATION_ANGLE = 90.0
SHOTS_PER_ENEMY = 1
SHOT_INTERVAL = 1.0
ENEMY_PAUSE_DURATION = 3.0  # 적 발견 시 3초 대기

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

async def analyze_obstacle(obstacle, index):
    x_center = (obstacle["x_min"] + obstacle["x_max"]) / 2
    z_center = (obstacle["z_min"] + obstacle["z_max"]) / 2
    image_data = obstacle.get("image")

    target_classes = {
        0: 'car002', 1: 'car003', 2: 'car005', 3: 'human1',
        4: 'rock1', 5: 'rock2', 6: 'tank', 7: 'wall1', 8: 'wall2'
    }

    class_name = 'unknown'
    confidence = 0.0

    if not image_data:
        print(f"🔍 YOLO: No image, classified as unknown at ({x_center:.2f}, {z_center:.2f})")
        logging.info(f"YOLO: No image, classified as unknown at ({x_center:.2f}, {z_center:.2f})")
        return {"className": class_name, "position": (x_center, z_center), "confidence": confidence}

    try:
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        cv_image = np.array(image)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

        results = model.predict(image, verbose=False)

        if len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None:
            detections = results[0].boxes.data.cpu().numpy()
        else:
            detections = np.array([])

        print(f"🔍 Raw YOLO detections: {detections}")

        filtered_results = [
            {'className': target_classes[int(box[5])], 'confidence': float(box[4]), 'bbox': box[:4]}
            for box in detections if int(box[5]) in target_classes
        ]

        if filtered_results:
            detection = max(filtered_results, key=lambda x: x['confidence'])
            class_name = detection['className']
            confidence = detection['confidence']
            bbox = detection['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            label = f"{class_name} {confidence:.2f}"

            print(f"🧩Detected class: {class_name}")
            print(f"🧩ENEMY_CLASSES: {ENEMY_CLASSES}")
            print(f"🧩Is enemy? {class_name in ENEMY_CLASSES}")
            print(f"🧩Is friendly? {class_name in FRIENDLY_CLASSES}")
            print(f"🧩Is obstacle? {class_name in OBSTACLE_CLASSES}")
            # ✅ 투명한 바운딩 박스 + 클래스명 표시 + 종류별 색깔 설정
            overlay = cv_image.copy()

            # 박스 색깔 지정
            if class_name in ENEMY_CLASSES:
                color = (0, 0, 255)  # 빨간색 (적)
            elif class_name in FRIENDLY_CLASSES:
                color = (255, 0, 0)  # 파란색 (아군)
            elif class_name in OBSTACLE_CLASSES:
                color = (0, 255, 0)  # 초록색 (장애물)
            else:
                color = (255, 255, 0)  # 노란색 (미분류)

            # 사각형 테두리만
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=2)

            # 라벨 텍스트
            cv2.putText(overlay, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # 투명도 적용
            alpha = 0.3
            cv_image = cv2.addWeighted(overlay, alpha, cv_image, 1 - alpha, 0)

            print(f"🔍 YOLO: {class_name} at ({x_center:.2f}, {z_center:.2f}), confidence={confidence:.2f}")
            logging.info(f"YOLO: {class_name} at ({x_center:.2f}, {z_center:.2f}), confidence={confidence:.2f}")
        else:
            print(f"🔍 YOLO: No valid detections at ({x_center:.2f}, {z_center:.2f})")
            logging.info(f"YOLO: No valid detections at ({x_center:.2f}, {z_center:.2f})")

    except Exception as e:
        print(f"🔍 YOLO failed: {e}")
        logging.error(f"YOLO failed: {e}")
        detections = np.array([])
    
    # 탐지 결과별 추가 출력
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

async def shoot_at_target(target_pos):
    with state_lock:
        current_time = asyncio.get_event_loop().time()
        player_state["last_shot_time"] = current_time - player_state["shot_cooldown"]
        player_state["last_shot_target"] = target_pos
        bullet_data = {"x": target_pos[0], "y": 0.0, "z": target_pos[1], "hit": "enemy"}
        for attempt in range(2):
            try:
                response = requests.post('http://localhost:5000/update_bullet', json=bullet_data, timeout=5)
                if response.status_code == 200:
                    player_state["shots_fired"] += 1
                    print(f"🔫 Shot #{player_state['shots_fired']} fired at {target_pos}")
                    logging.info(f"Shot #{player_state['shots_fired']} fired at {target_pos}")
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
        if current == goal_node:
            path = []
            while current in came_from:
                path.append((current.x, current.z))
                current = came_from[current]
            path.append((start_node.x, start_node.z))
            path.reverse()
            print(f"🛤️ A* path: {path}")
            logging.info(f"A* path: {path}")
            return path
        for neighbor in grid.get_neighbors(current):
            tentative_g_score = g_score[current] + (1.4 if abs(neighbor.x - current.x) + abs(neighbor.z - current.z) > 1 else 1)
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal_node)
                counter += 1
                heapq.heappush(open_set, (f_score[neighbor], tentative_g_score, counter, neighbor))
    print(f"🚫 A* failed: No path from ({start_node.x}, {start_node.z}) to ({goal_node.x}, {goal_node.z})")
    logging.error(f"A* failed: No path from ({start_node.x}, {start_node.z}) to ({goal_node.x}, {goal_node.z})")
    return []

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
            dest = player_state["destination"]
            state = player_state["state"]
            body_x = player_state["body_x"]
            current_time = asyncio.get_event_loop().time()
            if player_state["last_move_time"] is None:
                player_state["last_move_time"] = current_time
                player_state["last_move_position"] = current_pos
        distance = math.hypot(dest[0] - current_pos[0], dest[1] - current_pos[1])
        print(f"🚗 Pos={current_pos}, Dest={dest}, Distance={distance:.2f}, State={state}, BodyX={body_x:.2f}")
        logging.info(f"Pos={current_pos}, Dest={dest}, Distance={distance:.2f}, State={state}, BodyX={body_x:.2f}")

        # 목적지 근처 적 확인
        enemy_at_destination = False
        for idx, obstacle in enumerate(obstacles):
            obs_center = ((obstacle["x_min"] + obstacle["x_max"]) / 2, (obstacle["z_min"] + obstacle["z_max"]) / 2)
            dest_distance = math.hypot(obs_center[0] - dest[0], obs_center[1] - dest[1])
            if dest_distance < 5.0:  # 목적지 5m 이내
                detection = await analyze_obstacle(obstacle, idx)
                if detection["className"] in ENEMY_CLASSES and detection["confidence"] >= CONFIDENCE_THRESHOLD:
                    enemy_at_destination = True
                    print(f"🔫 Enemy at destination: {detection['className']} at {obs_center}, stopping at range")
                    logging.info(f"Enemy at destination: {detection['className']} at {obs_center}, stopping at range")
                    if distance <= DETECTION_RANGE:
                        with state_lock:
                            player_state["state"] = "PAUSE"
                            player_state["pause_start_time"] = current_time
                        print(f"⏸ Pausing for {ENEMY_PAUSE_DURATION}s before firing at destination enemy")
                        logging.info(f"Pausing for {ENEMY_PAUSE_DURATION}s before firing at destination enemy")
                        await asyncio.sleep(ENEMY_PAUSE_DURATION)
                        await shoot_at_target(obs_center)
                        print(f"🔫 Fired at destination enemy, re-planning path")
                        logging.info(f"Fired at destination enemy, re-planning path")
                        grid = Grid()  # 그리드 초기화
                        for obs in obstacles:
                            det = await analyze_obstacle(obs, 0)
                            if det["className"] in OBSTACLE_CLASSES:
                                grid.set_obstacle(
                                    obs["x_min"], obs["x_max"],
                                    obs["z_min"], obs["z_max"],
                                    start_pos=current_pos, goal_pos=dest
                                )
                    else:
                        # 사정거리 밖이면 계속 이동
                        pass
                    break

        if enemy_at_destination and distance <= DETECTION_RANGE:
            await asyncio.sleep(0.2)
            continue

        if distance <= 1.0 and not enemy_at_destination:
            with state_lock:
                player_state["state"] = "STOPPED"
                player_state["destination"] = None
                player_state["last_move_time"] = None
                player_state["last_move_position"] = None
            print(f"🎉 Arrived at destination: {dest}")
            logging.info(f"Arrived at destination: {dest}")
            break

        if state not in ["MOVING", "ESCAPING"]:
            await asyncio.sleep(0.2)
            continue

        # 갇힘 감지
        if state == "MOVING":
            enemy_found = False
            enemy_position = None

        for idx, obstacle in enumerate(obstacles):
            obs_center = ((obstacle["x_min"] + obstacle["x_max"]) / 2, (obstacle["z_min"] + obstacle["z_max"]) / 2)
            obs_distance = math.hypot(obs_center[0] - current_pos[0], obs_center[1] - current_pos[1])
            # 거리 초과여서 적탐지가 무시됏는지 확인
            print(f"🛰Player position: {current_pos}") 
            print(f"🛰Obstacle center: {obs_center}")
            print(f"🛰Distance to obstacle: {obs_distance:.2f}")

            if obs_distance < DETECTION_RANGE:
                detection = await analyze_obstacle(obstacle, idx)
    
                print(f"🧩 Detected object: {detection['className']} with confidence {detection['confidence']:.2f}")

                if detection["className"] in ENEMY_CLASSES and detection["confidence"] >= CONFIDENCE_THRESHOLD:
                    enemy_found = True
                    enemy_position = obs_center
                    break
        #Pause 상태에서 적 발견 시 포격
        if enemy_found:
            with state_lock:
                player_state["state"] = "PAUSE"
                player_state["pause_start_time"] = current_time
            print(f"⏸ Enemy nearby, pausing for {ENEMY_PAUSE_DURATION}s and firing")
            logging.info(f"Enemy nearby, pausing for {ENEMY_PAUSE_DURATION}s and firing")
            await asyncio.sleep(ENEMY_PAUSE_DURATION)
            await shoot_at_target(enemy_position)
            print("🔫 Shot fired at detected enemy, re-planning path")
            logging.info("Shot fired at detected enemy, re-planning path")

            grid = Grid()
            for obs in obstacles:
                det = await analyze_obstacle(obs, 0)
                if det["className"] in OBSTACLE_CLASSES:
                    grid.set_obstacle(
                        obs["x_min"], obs["x_max"],
                        obs["z_min"], obs["z_max"],
                        start_pos=current_pos, goal_pos=dest
                    )
            await asyncio.sleep(0.2)

            move_distance = math.hypot(current_pos[0] - player_state["last_move_position"][0],
                                      current_pos[1] - player_state["last_move_position"][1])
            if current_time - player_state["last_move_time"] > TRAPPED_TIMEOUT and move_distance < 0.5:
                with state_lock:
                    player_state["state"] = "ESCAPING"
                    player_state["escape_rotation_target"] = (player_state["body_x"] + ESCAPE_ROTATION_ANGLE) % 360
                    player_state["escape_start_time"] = current_time
                    player_state["last_move_time"] = current_time
                    player_state["last_move_position"] = current_pos
                print(f"🚫 Trapped detected: No movement for {TRAPPED_TIMEOUT}s at {current_pos}")
                print(f"🔄 Starting ESCAPING: Rotating to {player_state['escape_rotation_target']:.2f}°")
                logging.info(f"Trapped detected: No movement for {TRAPPED_TIMEOUT}s at {current_pos}")
                logging.info(f"Starting ESCAPING: Rotating to {player_state['escape_rotation_target']:.2f}°")
                await asyncio.sleep(0.2)
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
            angle_diff = (angle_to_next - body_x) % 360
            if angle_diff > 180:
                angle_diff -= 360
            if abs(angle_diff) > 135:
                print(f"🚫 Skipping waypoint {next_pos}: angle_diff={angle_diff:.2f}°, too far from body_x={body_x:.2f}")
                logging.warning(f"Skipping waypoint {next_pos}: angle_diff={angle_diff:.2f}°, too far from body_x={body_x:.2f}")
                await asyncio.sleep(0.2)
                continue
        with state_lock:
            player_state["position"] = next_pos
            player_state["last_move_time"] = current_time
            player_state["last_move_position"] = next_pos
        print(f"🚗 Moved to waypoint: {next_pos}, angle_to_next={angle_to_next:.2f}°")
        logging.info(f"Moved to waypoint: {next_pos}, angle_to_next={angle_to_next:.2f}°")

        if state == "MOVING":
            for idx, obstacle in enumerate(obstacles):
                obs_center = ((obstacle["x_min"] + obstacle["x_max"]) / 2, (obstacle["z_min"] + obstacle["z_max"]) / 2)
                obs_distance = math.hypot(obs_center[0] - current_pos[0], obs_center[1] - current_pos[1])
                if obs_distance < DETECTION_RANGE:
                    detection = await analyze_obstacle(obstacle, idx)
                    if detection["className"] in ENEMY_CLASSES and detection["confidence"] >= CONFIDENCE_THRESHOLD:
                        with state_lock:
                            player_state["state"] = "PAUSE"
                            player_state["pause_start_time"] = current_time
                        print(f"⏸ Pausing for {ENEMY_PAUSE_DURATION}s before firing at enemy")
                        logging.info(f"Pausing for {ENEMY_PAUSE_DURATION}s before firing at enemy")
                        await asyncio.sleep(ENEMY_PAUSE_DURATION)
                        await shoot_at_target(obs_center)
                        print(f"🔫 Fired at enemy, re-planning path")
                        logging.info(f"Fired at enemy, re-planning path")
                        grid = Grid()  # 그리드 초기화
                        for obs in obstacles:
                            det = await analyze_obstacle(obs, 0)
                            if det["className"] in OBSTACLE_CLASSES:
                                grid.set_obstacle(
                                    obs["x_min"], obs["x_max"],
                                    obs["z_min"], obs["z_max"],
                                    start_pos=current_pos, goal_pos=dest
                                )
                        break
                    elif detection["className"] in OBSTACLE_CLASSES and detection["confidence"] >= CONFIDENCE_THRESHOLD:
                        print(f"🪨 Obstacle detected: {detection['className']} at {obs_center}, distance={obs_distance:.2f}m")
                        logging.info(f"Obstacle detected: {detection['className']} at {obs_center}, distance={obs_distance:.2f}m")
                    else:
                        print(f"🔫 Skipped: Not an enemy or low confidence ({detection['className']}, confidence={detection['confidence']:.2f})")
                        logging.info(f"Skipped: Not an enemy or low confidence ({detection['className']}, confidence={detection['confidence']:.2f})")
        await asyncio.sleep(0.2)
    print("🏁 Movement stopped")
    logging.info("Movement stopped")

def check_obstacle_collision(x, z, obstacles):
    for obs in obstacles:
        # 1미터 버퍼 적용
        if (obs["x_min"] - 1.0) <= x <= (obs["x_max"] + 1.0) and (obs["z_min"] - 1.0) <= z <= (obs["z_max"] + 1.0):
            print(f"🚫 Collision at ({x:.2f}, {z:.2f})")
            logging.error(f"Collision at ({x:.2f}, {z:.2f})")
            return True
    return False

def run_async_task():
    retries = 3
    for attempt in range(retries):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(move_towards_destination())
            break
        except Exception as e:
            print(f"🚫 Async task failed (attempt {attempt+1}/{retries}): {e}, destination={player_state['destination']}")
            logging.error(f"Async task failed (attempt {attempt+1}/{retries}): {e}, destination={player_state['destination']}")
            if attempt < retries - 1:
                time.sleep(1)
        finally:
            loop.close()

@app.route('/detect', methods=['POST'])
def detect():
    image = request.files.get('image')
    print(image)
    if not image:
        print("Warning: YOLO: No image")
        logging.error("YOLO: No image")
        return jsonify({"error": "No image received"}), 400
    try:
        image = Image.open(image)
        results = model.predict(image, verbose=False)
        detections = results[0].boxes.data.cpu().numpy()
        print(f'🔍 Raw YOLO: {detections}')

        target_classes = {0: 'car002', 1: 'car003', 2: 'car005', 3: 'human1', 4: 'rock1', 5: 'rock2', 6: 'tank', 7: 'wall1', 8: 'wall2'}
        filtered_results = [
            {'className': target_classes[int(box[5])], 'bbox': [float(coord) for coord in box[:4]], 'confidence': float(box[4])}
            for box in detections if int(box[5]) in target_classes
        ]
        print(f"🔍 YOLO: {len(filtered_results)} detections")
        logging.info(f"YOLO: {len(filtered_results)} detections")
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
                    target = player_state.get("escape_rotation_target", expected_body_x if 'expected_body_x' in locals() else body_x)
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

    current_angle = math.radians(player_state["body_x"])
    if player_state["last_position"] and player_state["position"] != player_state["last_position"]:
        dx = player_state["position"][0] - player_state["last_position"][0]
        dz = player_state["position"][1] - player_state["last_position"][1]
        if math.hypot(dx, dz) > 1e-4:
            current_angle = math.atan2(dz, dx)
            player_state["last_valid_angle"] = current_angle
    elif player_state["last_valid_angle"] is None:
        dx, dz = player_state["destination"]
        px, pz = player_state["position"]
        player_state["last_valid_angle"] = math.atan2(dz - pz, dx - px)

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
            fx, fz = math.cos(current_angle), math.sin(current_angle)
            tx, tz = dx - px, dz - pz
            dist = math.hypot(tx, tz)
            if dist > 1e-6:
                tx /= dist
                tz /= dist
            dot = max(-1.0, min(1.0, fx*tx + fz*tz))
            angle_diff_rad = math.acos(dot)
            angle_diff_deg = math.degrees(angle_diff_rad)
            expected_body_x = math.degrees(math.atan2(tz, tx)) % 360
            angle_diff = (expected_body_x - player_state["body_x"]) % 360
            if angle_diff > 180:
                angle_diff -= 360
            control = "A" if angle_diff > 0 else "D"
            print(f"🧭 ROTATING: angle_diff={angle_diff_deg:.2f}°, shortest_angle={angle_diff:.2f}°, control={control}, expected_body_x={expected_body_x:.2f}")
            logging.info(f"ROTATING: angle_diff={angle_diff_deg:.2f}°, shortest_angle={angle_diff:.2f}°, control={control}, expected_body_x={expected_body_x:.2f}")
            logging.warning(f"Client should update body_x to {expected_body_x:.2f}")
            if abs(player_state["body_x"] - expected_body_x) < ROTATION_THRESHOLD_DEG:
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
            angle_diff = (expected_body_x - player_state["body_x"]) % 360
            if angle_diff > 180:
                angle_diff -= 360
            control = "A" if angle_diff > 0 else "D"
            angle_diff_deg = abs(angle_diff)
            print(f"🔄 ESCAPING: angle_diff={angle_diff_deg:.2f}°, control={control}, expected_body_x={expected_body_x:.2f}")
            logging.info(f"ESCAPING: angle_diff={angle_diff_deg:.2f}°, control={control}, expected_body_x={expected_body_x:.2f}")
            logging.warning(f"Client should update body_x to {expected_body_x:.2f}")
            if abs(player_state["body_x"] - expected_body_x) < ROTATION_THRESHOLD_DEG:
                player_state["state"] = "PAUSE"
                player_state["pause_start_time"] = current_time
                player_state["escape_rotation_target"] = None
                player_state["escape_start_time"] = None
                global grid
                grid = Grid()
                print(f"✅ ESCAPING completed: body_x={player_state['body_x']:.2f}")
                print(f"🔄 ESCAPING -> PAUSE: body_x={player_state['body_x']:.2f}, expected_body_x={expected_body_x:.2f}")
                logging.info(f"ESCAPING completed: body_x={player_state['body_x']:.2f}")
                logging.info(f"ESCAPING -> PAUSE: body_x={player_state['body_x']:.2f}, expected_body_x={expected_body_x:.2f}")
            elif player_state["escape_start_time"] and (current_time - player_state["escape_start_time"]) > ROTATION_TIMEOUT:
                player_state["state"] = "PAUSE"
                player_state["pause_start_time"] = current_time
                player_state["escape_rotation_target"] = None
                player_state["escape_start_time"] = None
                grid = Grid()
                print(f"✅ ESCAPING completed: body_x={player_state['body_x']:.2f} (timeout)")
                print(f"🔄 ESCAPING -> PAUSE: timeout")
                logging.info(f"ESCAPING completed: body_x={player_state['body_x']:.2f} (timeout)")
                logging.info(f"ESCAPING -> PAUSE: timeout")
            else:
                weight = calculate_rotation_weight(angle_diff_deg)

        elif player_state["state"] == "PAUSE":
            if (current_time - player_state["pause_start_time"]) >= PAUSE_DURATION:
                player_state["state"] = "MOVING"
                control = "W"
                weight = calculate_move_weight(player_state["distance_to_destination"])
                threading.Thread(target=run_async_task, daemon=True).start()
                print(f"🔄 PAUSE -> MOVING, control={control}, weight={weight:.2f}, duration={PAUSE_DURATION}s")
                logging.info(f"PAUSE -> MOVING, control={control}, weight={weight:.2f}, duration={PAUSE_DURATION}s")
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
                fx, fz = math.cos(current_angle), math.sin(current_angle)
                tx, tz = dx - px, dz - pz
                dist = math.hypot(tx, tz)
                if dist > 1e-6:
                    tx /= dist
                    tz /= dist
                dot = max(-1.0, min(1.0, fx*tx + fz*tz))
                angle_diff_rad = math.acos(dot)
                angle_diff_deg = math.degrees(angle_diff_rad)
                expected_body_x = math.degrees(math.atan2(tz, tx)) % 360
                angle_diff = (expected_body_x - player_state["body_x"]) % 360
                if angle_diff > 180:
                    angle_diff -= 360
                if angle_diff_deg > 45:
                    player_state["state"] = "ROTATING"
                    player_state["rotation_start_time"] = current_time
                    control = "A" if angle_diff > 0 else "D"
                    weight = calculate_rotation_weight(angle_diff_deg)
                    angle_to_dest = math.degrees(math.atan2(tz, tx)) % 360
                    print(f"🔄 MOVING -> ROTATING: angle_to_dest={angle_to_dest:.2f}°, body_x={player_state['body_x']:.2f}, angle_diff={angle_diff:.2f}°")
                    logging.info(f"MOVING -> ROTATING: angle_to_dest={angle_to_dest:.2f}°, body_x={player_state['body_x']:.2f}, angle_diff={angle_diff:.2f}°")
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
        if player_state["state"] == "STOPPED":
            return jsonify({"move": "STOP", "weight": 0.0})
        elif player_state["state"] == "MOVING":
            weight = calculate_move_weight(player_state["distance_to_destination"])
            return jsonify({"move": "W", "weight": weight})
        elif player_state["state"] in ["ROTATING", "ESCAPING"]:
            return jsonify({"move": "A", "weight": 0.3})
        elif player_state["state"] == "PAUSE":
            return jsonify({"move": "STOP", "weight": 0.0})
        else:
            return jsonify({"move": "STOP", "weight": 0.0})

@app.route('/get_action', methods=['GET'])
def get_action():
    with state_lock:
        if player_state["enemy_detected"] and player_state["state"] == "MOVING":
            print(f"🔫 FIRE: target={player_state['last_shot_target']}")
            logging.info(f"FIRE: target={player_state['last_shot_target']}")
            player_state["enemy_detected"] = False
            return jsonify({"turret": "FIRE", "weight": 1.0})
        print("🔫 No enemy detected or not in MOVING state")
        logging.info("No enemy detected or not in MOVING state")
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
    grid = Grid()
    with state_lock:
        start_pos = player_state["position"]
        goal_pos = player_state["destination"]
    for obstacle in obstacles:
        detection = asyncio.run(analyze_obstacle(obstacle, 0))
        if detection["className"] in OBSTACLE_CLASSES:
            grid.set_obstacle(
                obstacle["x_min"], obstacle["x_max"],
                obstacle["z_min"], obstacle["z_max"],
                start_pos=start_pos, goal_pos=goal_pos
            )
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