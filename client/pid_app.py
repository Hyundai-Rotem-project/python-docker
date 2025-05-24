from flask import Flask, request, jsonify
import os
import time
import json
import torch
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pyautogui
import atexit

app = Flask(__name__)
model = YOLO('best.pt')

IMAGE_WIDTH = 2560 # Unity에서 보내는 이미지 해상도에 맞춰 수정
IMAGE_HEIGHT = 1577 # Unity에서 보내는 이미지 해상도에 맞춰 수정
IMAGE_CENTER_X = IMAGE_WIDTH / 2
IMAGE_CENTER_Y = IMAGE_HEIGHT / 2

latest_enemy_list = []
tracking_enabled = True
auto_fire_log = []

# PID 제어 변수 (여기서는 PID 피드백을 통해 키를 누를지 말지만 결정하고, 실제 '누르는' 횟수는 제어하지 않음)
pid_yaw = {
    "kp": 0.1, "ki": 0.01, "kd": 0.05,
    "integral": 0.0, "last_error": 0.0, "last_time": time.time(),
    "log": []
}

pid_pitch = {
    "kp": 0.1, "ki": 0.01, "kd": 0.05,
    "integral": 0.0, "last_error": 0.0, "last_time": time.time(),
    "log": []
}

GLOBAL_WEIGHT = 0.1
AUTO_GRAPH_INTERVAL = 10

# 키를 빠르게 누를 간격 (이 값을 조정하여 포신 움직임의 부드러움을 조절)
# 너무 짧으면 CPU 부하가 커지고, 너무 길면 움직임이 끊깁니다.
KEY_PRESS_INTERVAL = 0.05 # 50ms마다 키를 누름

# --- 유틸리티 함수 ---

def compute_pid(pid, error):
    now = time.time()
    dt = now - pid["last_time"]
    pid["last_time"] = now

    if dt < 0.0001:
        return pid["kp"] * error + pid["ki"] * pid["integral"] + pid["kd"] * 0.0

    pid["integral"] += error * dt
    derivative = (error - pid["last_error"]) / dt
    pid["last_error"] = error

    feedback = pid["kp"] * error + pid["ki"] * pid["integral"] + pid["kd"] * derivative
    pid["log"].append((time.time(), error, feedback))

    max_output = 1.0
    feedback = max(min(feedback, max_output), -max_output)
    return feedback

def save_pid_graph():
    for name, pid in [("Yaw", pid_yaw), ("Pitch", pid_pitch)]:
        if not pid["log"]:
            continue

        times = [x[0] for x in pid["log"]]
        errors = [x[1] for x in pid["log"]]
        feedbacks = [x[2] for x in pid["log"]]

        start_time = times[0]
        relative_times = [t - start_time for t in times]

        plt.figure(figsize=(10, 4))
        plt.plot(relative_times, errors, label=f"{name} Error")
        plt.plot(relative_times, feedbacks, label=f"{name} PID Output")
        plt.xlabel("Time (s)")
        plt.ylabel("Value")
        plt.title(f"PID Control - {name}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"pid_{name.lower()}_log.png")
        plt.close()

# qqq eee 방식에서는 키를 '누르고 떼는' 상태 관리가 불필요하므로 이 함수는 더 이상 사용되지 않음
# 하지만 혹시 모를 경우를 대비하여 껍데기만 남겨둠.
def release_all_keys():
    print("release_all_keys() is not actively used in 'qqq eee' mode.")
    pass

# --- Flask 라우트 핸들러 ---

@app.route('/detect', methods=['POST'])
def detect():
    global latest_enemy_list
    image = request.files.get('image')
    if not image:
        return jsonify({"error": "No image received"}), 400

    image_path = 'temp_image.jpg'
    image.save(image_path)

    results = model(image_path)
    detections = results[0].boxes.data.cpu().numpy()

    target_classes = {0: 'car2', 1: 'car3', 2: 'car5', 3: 'human', 4: 'rock', 5: 'tank', 6: 'wall'}
    latest_enemy_list = []

    for box in detections:
        class_id = int(box[5])
        if class_id in target_classes:
            latest_enemy_list.append({
                'className': target_classes[class_id],
                'bbox': [float(coord) for coord in box[:4]],
                'confidence': float(box[4])
            })

    import cv2
    img = cv2.imread(image_path)
    for box in latest_enemy_list:
        x1, y1, x2, y2 = map(int, box['bbox'])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, box['className'], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imwrite("last_detection.jpg", img)

    return jsonify(latest_enemy_list)

@app.route('/get_action', methods=['POST'])
def get_action():
    global latest_enemy_list, pid_yaw, pid_pitch, auto_fire_log, tracking_enabled

    # 트래킹 모드가 꺼져 있거나 적이 없으면 키 입력 중지
    if not tracking_enabled or not latest_enemy_list:
        # qqq eee 방식에서는 지속적으로 키를 누르는 것이 아니므로,
        # 이전에 눌렀던 키를 해제하는 로직은 불필요합니다.
        # 즉, 키 입력 명령을 보내지 않으면 자동으로 멈춥니다.
        return jsonify({
            "moveWS": {"command": "STOP", "weight": 1.0},
            "moveAD": {"command": "", "weight": 0.0},
            "turretQE": {"command": "", "weight": 0.0},
            "turretRF": {"command": "", "weight": 0.0},
            "fire": False
        })

    enemy = latest_enemy_list[0]
    bbox = enemy.get("bbox", [0, 0, 0, 0])
    bbox_center_x = (bbox[0] + bbox[2]) / 2
    bbox_center_y = (bbox[1] + bbox[3]) / 2

    err_x = bbox_center_x - IMAGE_CENTER_X
    err_y = bbox_center_y - IMAGE_CENTER_Y

    feedback_x = compute_pid(pid_yaw, err_x)
    feedback_y = compute_pid(pid_pitch, err_y)

    DEADZONE = 0.05 # 이 값은 포신 움직임의 민감도를 조절합니다. (0.05는 매우 민감)

    # ------------------- 포신 Yaw (수평) 제어 -------------------
    target_yaw_command = ""
    if feedback_x < -DEADZONE:
        target_yaw_command = "Q" # "left" 키를 누름
    elif feedback_x > DEADZONE:
        target_yaw_command = "E" # "right" 키를 누름

    if target_yaw_command != "":
        # PID 피드백이 양수이면 "right"를, 음수이면 "left"를 빠르게 누름
        # pyautogui.press()는 키를 누르고 떼는 동작을 수행합니다.
        # KEY_PRESS_INTERVAL 시간 동안 잠시 대기
        pyautogui.press(target_yaw_command, interval=KEY_PRESS_INTERVAL)

    # ------------------- 포신 Pitch (수직) 제어 -------------------
    target_pitch_command = ""
    if feedback_y < -DEADZONE:
        target_pitch_command = "R" # "up" 키를 누름
    elif feedback_y > DEADZONE:
        target_pitch_command = "F" # "down" 키를 누름

    if target_pitch_command != "":
        pyautogui.press(target_pitch_command, interval=KEY_PRESS_INTERVAL)


    # 발사 조건 (오차 5.0 픽셀 이내일 때 발사, 이 값도 튜닝 가능)
    fire = abs(err_x) < 5.0 and abs(err_y) < 5.0

    print("📦 Latest enemy list:", latest_enemy_list)
    print(f"🎯 err_x: {err_x:.2f}, err_y: {err_y:.2f}")
    print(f"🔁 Yaw Command: '{target_yaw_command}', Pitch Command: '{target_pitch_command}'")
    print(f"🔥 fire: {fire}")

    if fire:
        auto_fire_log.append({"time": time.time(), "bbox": bbox})
        if len(auto_fire_log) % AUTO_GRAPH_INTERVAL == 0:
            save_pid_graph()
            print("📊 PID 그래프 자동 저장 완료.")
        pyautogui.press("space")

    if target_yaw_command:
        command = target_yaw_command.pop(0)
        return jsonify({command})

# --- 기타 Unity 통신 관련 라우트 (기존 코드 유지) ---

@app.route('/info', methods=['POST'])
def info():
    data = request.get_json(force=True)
    return jsonify({"status": "success", "control": ""})

@app.route('/update_bullet', methods=['POST'])
def update_bullet():
    data = request.get_json()
    print(f"💥 Bullet Impact: {data}")
    return jsonify({"status": "OK", "message": "Bullet impact data received"})

@app.route('/set_destination', methods=['POST'])
def set_destination():
    data = request.get_json()
    try:
        x, y, z = map(float, data["destination"].split(","))
        return jsonify({"status": "OK", "destination": {"x": x, "y": y, "z": z}})
    except Exception as e:
        return jsonify({"status": "ERROR", "message": str(e)}), 400

@app.route('/update_obstacle', methods=['POST'])
def update_obstacle():
    data = request.get_json()
    print("🪨 Obstacle Data:", data)
    return jsonify({'status': 'success', 'message': 'Obstacle data received'})

@app.route('/collision', methods=['POST'])
def collision():
    data = request.get_json()
    object_name = data.get('objectName')
    position = data.get('position', {})
    x = position.get('x')
    y = position.get('y')
    z = position.get('z')

    print(f"💥 Collision Detected - Object: {object_name}, Position: ({x}, {y}, {z})")
    return jsonify({'status': 'success', 'message': 'Collision data received'})

@app.route('/init', methods=['GET'])
def init():
    config = {
        "startMode": "start",
        "blStartX": 60, "blStartY": 10, "blStartZ": 27.23,
        "rdStartX": 59, "rdStartY": 10, "rdStartZ": 280,
        "trackingMode": True, "detactMode": False,
        "logMode": True, "enemyTracking": True,
        "saveSnapshot": False, "saveLog": True,
        "saveLidarData": False, "lux": 30000
    }
    print("🛠️ Initialization:", config)
    return jsonify(config)

@app.route('/start', methods=['GET'])
def start():
    print("🚀 Start command received")
    try:
        unity_windows = pyautogui.getWindowsWithTitle("Unity")
        if unity_windows:
            unity_windows[0].activate()
            print("Unity window activated.")
        else:
            print("Unity window not found. Ensure it's running and focused at least once.")
    except Exception as e:
        print(f"Error activating Unity window: {e}")
    return jsonify({"control": ""})

@app.route('/set_tracking', methods=['POST'])
def set_tracking():
    global tracking_enabled
    data = request.get_json()
    if data is not None and "enable" in data:
        tracking_enabled = bool(data["enable"])
        # qqq eee 방식에서는 release_all_keys()가 필요 없지만, 기존 구조 유지를 위해.
        # if not tracking_enabled:
        #    release_all_keys()
        print(f"🎯 Tracking mode explicitly set to: {tracking_enabled}")
        return jsonify({"status": "OK", "tracking": tracking_enabled})
    else:
        return jsonify({"status": "ERROR", "message": "Missing 'enable' field"}), 400

@app.route('/status', methods=['GET'])
def status():
    global tracking_enabled
    return jsonify({"tracking": tracking_enabled})

if __name__ == '__main__':
    # qqq eee 방식에서는 프로그램 종료 시 키 해제가 불필요하지만, 기존 구조 유지를 위해.
    # atexit.register(release_all_keys)
    app.run(host='0.0.0.0', port=5000)