from flask import Flask, request, jsonify
import math
import time
import torch
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO('best.pt')

# ── 전역 변수 ──
destination = None
current_position = None
last_position = None
last_valid_angle = None
state = "IDLE"                 # FSM 상태
distance_to_destination = float("inf")
rotation_start_time = None
pause_start_time = None
last_body_x = last_body_y = last_body_z = None

# /info 에서 계산된 최근 제어값
last_control = "STOP"
last_weight  = 0.0

# ── 상수 ──
ROTATION_THRESHOLD_DEG = 5    # 회전 완료 기준 (°)
STOP_DISTANCE = 45.0          # 정지 거리 (m)
SLOWDOWN_DISTANCE = 100.0     # 감속 시작 거리 (m)
ROTATION_TIMEOUT = 0.8        # 회전 최대 시간 (s)
PAUSE_DURATION = 0.5          # 회전 후 일시정지 (s)
WEIGHT_LEVELS = [1.0, 0.6, 0.3, 0.1, 0.05, 0.01]

def select_weight(value, levels=WEIGHT_LEVELS):
    return min(levels, key=lambda x: abs(x - value))

def calculate_move_weight(distance):
    if distance <= STOP_DISTANCE:
        return 0.0
    if distance > SLOWDOWN_DISTANCE:
        return 1.0
    norm = (distance - STOP_DISTANCE) / (SLOWDOWN_DISTANCE - STOP_DISTANCE)
    target = 0.01 + (1.0 - 0.01) * (norm ** 2)
    return select_weight(target)

def calculate_rotation_weight(angle_deg):
    if abs(angle_deg) < ROTATION_THRESHOLD_DEG:
        return 0.0
    target = min(0.3, (abs(angle_deg) / 45) * 0.3)
    return select_weight(target)

@app.route('/detect', methods=['POST'])
def detect():
    image = request.files.get('image')
    if not image:
        return jsonify({'error': 'No image received'}), 400
    path = 'temp_image.jpg'
    image.save(path)
    results = model(path)
    boxes = results[0].boxes.data.cpu().numpy()
    target_classes = {0: 'person', 2: 'car', 7: 'truck', 15: 'rock'}
    out = []
    for b in boxes:
        cid = int(b[5])
        if cid in target_classes:
            out.append({
                'className': target_classes[cid],
                'bbox': [float(c) for c in b[:4]],
                'confidence': float(b[4])
            })
    return jsonify(out)

@app.route('/info', methods=['POST'])
def info():
    global state, destination, current_position, last_position, distance_to_destination
    global rotation_start_time, pause_start_time, last_valid_angle
    global last_body_x, last_body_y, last_body_z, last_control, last_weight

    data = request.get_json(force=True)
    if not data:
        return jsonify({'error': 'No JSON received'}), 400

    # 목적지가 설정되지 않았다면 정지
    if not destination:
        state = "IDLE"
        last_control, last_weight = "STOP", 0.0
        return jsonify(status="success", control="STOP", weight=0.0)

    # 1) 입력 파싱
    p = data.get('playerPos', {})
    bodyX = data.get('playerBodyX', 0.0)
    bodyY = data.get('playerBodyY', 0.0)
    bodyZ = data.get('playerBodyZ', 0.0)
    distance_to_destination = data.get('distance', float("inf"))
    current_position = (p.get('x', 0.0), p.get('z', 0.0))

    # 2) 초기 방향 보정
    if last_position and current_position != last_position:
        dx = current_position[0] - last_position[0]
        dz = current_position[1] - last_position[1]
        if math.hypot(dx, dz) > 1e-4:
            current_angle = math.atan2(dz, dx)
        else:
            current_angle = math.radians(bodyX)
    else:
        dx, dz = destination
        px, pz = current_position
        current_angle = math.atan2(dz - pz, dx - px)
    last_valid_angle = current_angle

    # 3) 바디 방향 변화 로그
    if last_body_x is not None:
        dbx, dby, dbz = bodyX - last_body_x, bodyY - last_body_y, bodyZ - last_body_z
        if abs(dbx) < 1e-3 and state == "ROTATING":
            print("⚠️ bodyX change too small during ROTATING")
        print(f"🔄 Δbody: X={dbx:.3f}, Y={dby:.3f}, Z={dbz:.3f}")
    last_body_x, last_body_y, last_body_z = bodyX, bodyY, bodyZ

    # 4) FSM 처리
    control, weight = "STOP", 0.0

    if state == "IDLE":
        state = "ROTATING"
        rotation_start_time = time.time()

    elif state == "ROTATING":
        dx, dz = destination
        px, pz = current_position

        # 현재 전방 벡터
        fx, fz = math.cos(current_angle), math.sin(current_angle)
        # 목표 방향 벡터 (정규화)
        tx, tz = dx - px, dz - pz
        dist = math.hypot(tx, tz)
        if dist > 1e-6:
            tx /= dist
            tz /= dist

        # 내적으로 각도 차이 계산
        dot = max(-1.0, min(1.0, fx*tx + fz*tz))
        angle_diff_rad = math.acos(dot)
        deg = math.degrees(angle_diff_rad)
        # 외적(z 성분)으로 회전 방향 판별
        cross = fx * tz - fz * tx

        print(f"🧭 ROTATING: angle_diff={deg:.2f}°, cross={cross:.3f}")

        # 회전 타임아웃 또는 완료 판정
        if rotation_start_time and (time.time() - rotation_start_time) > ROTATION_TIMEOUT:
            state = "PAUSE"
            pause_start_time = time.time()
        elif deg < ROTATION_THRESHOLD_DEG:
            state = "PAUSE"
            pause_start_time = time.time()
        else:
            control = "A" if cross > 0 else "D"
            weight = calculate_rotation_weight(deg)

    elif state == "PAUSE":
        if (time.time() - pause_start_time) >= PAUSE_DURATION:
            state = "MOVING"
            control = "W"
            weight = calculate_move_weight(distance_to_destination)

    elif state == "MOVING":
        dx, dz = destination
        px, pz = current_position
        z_diff = abs(pz - dz)

        # 방향 재판단에도 동일한 벡터 로직 사용
        fx, fz = math.cos(current_angle), math.sin(current_angle)
        tx, tz = dx - px, dz - pz
        dist = math.hypot(tx, tz)
        if dist > 1e-6:
            tx /= dist
            tz /= dist
        dot = max(-1.0, min(1.0, fx*tx + fz*tz))
        angle_diff_rad = math.acos(dot)
        deg = math.degrees(angle_diff_rad)
        cross = fx * tz - fz * tx

        # 도착 조건
        if distance_to_destination <= STOP_DISTANCE or z_diff < 5.0:
            state = "STOPPED"
        # 큰 방향 오류 시 재회전
        elif abs(deg) > ROTATION_THRESHOLD_DEG * 6:
            state = "ROTATING"
            rotation_start_time = time.time()
            control = "A" if cross > 0 else "D"
            weight  = calculate_rotation_weight(deg)
        else:
            control = "W"
            weight  = calculate_move_weight(distance_to_destination)

    else:  # STOPPED
        control, weight = "STOP", 0.0

    # 5) 결과 저장 및 반환
    last_control, last_weight = control, weight
    last_position = current_position

    return jsonify(status="success", control=control, weight=weight)

@app.route('/set_destination', methods=['POST'])
def set_destination():
    global destination, state, rotation_start_time, last_position, last_valid_angle
    data = request.get_json()
    if not data or "destination" not in data:
        return jsonify({'status': 'ERROR', 'message': 'Missing destination'}), 400
    try:
        x, y, z = map(float, data["destination"].split(","))
        destination = (x, z)
        # 초기 방향 보정을 위해 리셋
        last_position = None
        last_valid_angle = None

        state = "ROTATING"
        rotation_start_time = time.time()
        print(f"🎯 New destination: {x},{y},{z} (reset last_position)")
        return jsonify(status="OK", destination={'x': x, 'y': y, 'z': z})
    except Exception as e:
        return jsonify({'status': 'ERROR', 'message': str(e)}), 400

@app.route('/update_position', methods=['POST'])
def update_position():
    global current_position, last_position, state, destination
    data = request.get_json()
    if not data or "position" not in data:
        return jsonify({'status': 'ERROR', 'message': 'Missing position data'}), 400
    try:
        x, y, z = map(float, data["position"].split(","))
        current_position = (x, z)
        if last_position:
            dx, dz = x - last_position[0], z - last_position[1]
            print(f"📍 Movement change: dx={dx:.6f}, dz={dz:.6f}")
        if destination:
            dx, dz = destination
            z_diff = abs(z - dz)
            direction_angle = math.degrees(math.atan2(dz - z, dx - x))
            print(f"📍 Position updated: {current_position}, target angle: {direction_angle:.2f}°, z_diff: {z_diff:.2f}m")
        else:
            print(f"📍 Position updated: {current_position}")
        return jsonify(status="OK", current_position=current_position)
    except Exception as e:
        return jsonify({'status': 'ERROR', 'message': str(e)}), 400

@app.route('/get_move', methods=['GET'])
def get_move():
    # /info에서 계산된 제어값을 그대로 반환
    return jsonify(move=last_control, weight=last_weight)

@app.route('/get_action', methods=['GET'])
def get_action():
    if state == "STOPPED":
        return jsonify(turret="FIRE", weight=1.0)
    return jsonify(turret="", weight=0.0)

@app.route('/update_bullet', methods=['POST'])
def update_bullet():
    data = request.get_json()
    if not data:
        return jsonify({'status': 'ERROR', 'message': 'Invalid data'}), 400
    print(f"💥 Bullet Impact at X={data.get('x')}, Y={data.get('y')}, Z={data.get('z')}, hit={data.get('hit')}")
    return jsonify(status="OK", message="Bullet impact data received")

@app.route('/update_obstacle', methods=['POST'])
def update_obstacle():
    data = request.get_json()
    if not data:
        return jsonify({'status': 'error', 'message': 'No data received'}), 400
    print("🪨 Obstacle Data:", data)
    return jsonify(status="success", message="Obstacle data received")

@app.route('/init', methods=['GET'])
def init():
    config = {
        "startMode": "start",
        "blStartX": 60, "blStartY": 10, "blStartZ": 27.23,
        "rdStartX": 59, "rdStartY": 10, "rdStartZ": 280
    }
    print("🛠️ Initialization config:", config)
    return jsonify(config)

@app.route('/start', methods=['GET'])
def start():
    print("🚀 /start command received")
    return jsonify(control="")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
