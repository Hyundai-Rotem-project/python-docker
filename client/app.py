# app.py
from flask import Flask, request, jsonify
import math
import time
import threading
from grid import Grid
from detection import detect
from movement import run_async_task
from config import (
    ROTATION_THRESHOLD_DEG, STOP_DISTANCE, SLOWDOWN_DISTANCE,
    ROTATION_TIMEOUT, PAUSE_DURATION, WEIGHT_LEVELS, OBSTACLE_CLASSES
)

app = Flask(__name__)

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
    "last_shot_target": None
}
state_lock = threading.Lock()

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

@app.route('/info', methods=['POST'])
def info():
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
                threading.Thread(target=run_async_task, args=(obstacles, grid, player_state, state_lock), daemon=True).start()
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
        from detection import analyze_obstacle
        detection = asyncio.run(analyze_obstacle(obstacle, 0))
        if detection["className"] in OBSTACLE_CLASSES:
            grid.set_obstacle(obstacle["x_min"], obstacle["x_max"], obstacle["z_min"], obstacle["z_max"])
    print("🪨 Obstacle Data:", obstacles)
    return jsonify({'status': 'success', 'message': 'Obstacle data received'})

@app.route('/detect', methods=['POST'])
def detect_endpoint():
    return detect(request.files.get('image'))

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