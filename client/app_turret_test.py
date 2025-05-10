from flask import Flask, request, jsonify, render_template
import logging
from flask_socketio import SocketIO
from PIL import Image, ImageDraw, ImageFont
import torch
from ultralytics import YOLO
import time
import json
import modules.turret as turret
import modules.is_near_enemy as is_near_enemy
import modules.get_enemy_pos as get_enemy_pos
import math

app = Flask(__name__)

DEBUG = True
STATE_DEBUG = True
 

# YOLO 모델 로드
try:
    model = YOLO('best.pt')

except Exception as e:
    raise RuntimeError(f"YOLO model loading failed: {str(e)}")


EXCLUDE_PATHS = ("/info", "/start", "/update_position", "/get_move", "/get_action")
class PathFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return not any(path in msg for path in EXCLUDE_PATHS)
log = logging.getLogger("werkzeug")
log.addFilter(PathFilter())

socketio = SocketIO(app)

# 전역 변수
move_command = []
action_command = []
player_data = {'pos': {'x': 60, 'y': 10, 'z': 57}}  # 기본 위치 설정
destination = {}
impact_info = {}
obstacles = []  # /set_obstacles 데이터 저장
latest_nearest_enemy = None
MATCH_THRESHOLD = 3.0

#정적인 적 - 가까운 적을 타격한 것을 표시하고 더이상 쏘지 않게 한다.
dead_list =[]

#3 FOV 및 카메라 설정
FOV_HORIZONTAL = 50
FOV_VERTICAL = 28
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
MAP_WIDTH = 300
MAP_HEIGHT = 300

@app.route('/dashboard')
def dashboard():
    if DEBUG: print('🚨 dashboard >>>')
    return render_template('dashboard.html')

first_action_state = True
hit_state = -1
@app.route('/detect', methods=['POST'])
def detect():
    global player_data, obstacles, latest_nearest_enemy, action_command, destination, first_action_state, hit_state
    print('🌍 detect >>>')

    # 1. 이미지 수신
    image = request.files.get('image')
    if not image:
        return jsonify({"error": "No image received"}), 400

    image_path = 'temp_image.jpg'
    image.save(image_path)

    results = model(image_path)
    detections = results[0].boxes.data.cpu().numpy()

    # 3. 탐지 결과 필터링
    target_classes = {
        0: 'car002', 1: 'car003', 2: 'car005', 3: 'human001',
        4: 'rock001', 5: 'rock2', 6: 'tank', 7: 'wall001', 8: 'wall002'
    }
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", size=20)
    except:
        font = ImageFont.load_default()

    print("Player position:", player_data['pos'])
    
    filtered_results = []
    for box in detections:
        class_id = int(box[5])
        if class_id in target_classes:
            filtered_results.append({
                'className': target_classes[class_id],
                'bbox': [float(coord) for coord in box[:4]],
                'confidence': float(box[4]),
                'color': '#00FF00',
                'filled': False,
                'updateBoxWhileMoving': False
            })

    return jsonify(filtered_results)
    # # 1. 이미지 수신
    # image = request.files.get('image')
    # if not image:
    #     return jsonify([])

    # image_path = 'temp_image.jpg'

    # try:
    #     image.save(image_path)
    # except Exception as e:
    #     return jsonify([])
    
    # # 2. 이미지 크기 확인
    # img = Image.open(image_path).convert('RGB')
    # width, height = img.size
    # print(f"이미지 크기: 너비 = {width} 픽셀, 높이 = {height} 픽셀")
    # logging.debug(f"Image size: width={width}px, height={height}px")

    # #3 FOV 및 카메라 설정
    # fov_horizontal = 50
    # fov_vertical = 28
    # player_pose = player_data.get('pose', {'x': 60, 'y': 8, 'z': 57})
    # map_width = 300
    # map_height = 300
    # print(f"FOV: 수평 = {fov_horizontal:.2f}도, 수직 = {fov_vertical:.2f}도")
    # print(f"카메라 위치: x = {player_pose['x']:.2f}m, y = {player_pose['y']:.2f}m, z = {player_pose['z']:.2f}m")
    # logging.debug(f"FOV: horizontal={fov_horizontal}°, vertical={fov_vertical}°")
    # logging.debug(f"Camera pose: {player_pose}")

    if STATE_DEBUG : print('1 🤩🤩first_action_state', first_action_state)
    if STATE_DEBUG : print('1 🤩🤩hit_state', hit_state)

    # 수정필요: 이동이 완전히 멈춘 상태가 되면 -> is_near_enemy.find_nearest_enemy 호출 (state 필요)
    nearest_enemy = get_enemy_pos.find_nearest_enemy(filtered_results, player_data, obstacles)

    if nearest_enemy['state'] and first_action_state:
        try:
            # if DEBUG: print(f"👉 Generating action command: player_pos={player_data.get('pos')}, dest={destination}")
            latest_nearest_enemy = nearest_enemy
            action_command = turret.get_action_command(
                player_data.get('pos', {'x': 60, 'y': 10, 'z': 57}),
                nearest_enemy,
                turret_x_angle=player_data.get('turret_x', 0),
                turret_y_angle=player_data.get('turret_y', 0),
                player_y_angle=player_data.get('body_y', 0)
            )
            print('🐟action_command', action_command)
            first_action_state = False
        except ValueError as e:
            print(f"🚫 Error generating action command: {str(e)}")
            action_command = []
        
        if STATE_DEBUG : print('2 🤩🤩action - first_action_state f', first_action_state)
        if STATE_DEBUG : print('2 🤩🤩action - hit_state -1', hit_state)

    # 💣 dead_list에 있는 적이면 무시
    if nearest_enemy['state']:
        ex = nearest_enemy['x']
        ez = nearest_enemy['z']
        if get_enemy_pos.is_already_dead(ex, ez, dead_list):
            print("🧟‍♂️ 이미 사망한 타겟. 포격 제외.")
            return jsonify({"status": "already_dead", "target": None})
    return jsonify(filtered_results)

@app.route('/info', methods=['POST'])
def info():
    if DEBUG: print('🚨 info >>>')
    global player_data
    data = request.get_json(force=True)
    if not data:
        if DEBUG: print("🚫 No JSON received")
        return jsonify({"error": "No JSON received", "control": ""}), 400

    player_data = {
        'pos': {
            'x': data.get('playerPos', {}).get('x', player_data.get('pos', {}).get('x', 60)),
            'y': data.get('playerPos', {}).get('y', player_data.get('pos', {}).get('y', 10)),
            'z': data.get('playerPos', {}).get('z', player_data.get('pos', {}).get('z', 57)),
        },
        'turret_x': data.get('playerTurretX', player_data.get('turret_x', 0)),
        'turret_y': data.get('playerTurretY', player_data.get('turret_y', 0)),
        'body_x': data.get('playerBodyX', player_data.get('body_x', 0)),
        'body_y': data.get('playerBodyY', player_data.get('body_y', 0)),
        'body_z': data.get('playerBodyZ', player_data.get('body_z', 0)),
    }
    # if DEBUG: print(f"📍 Player data updated: {player_data}")
    return jsonify({"status": "success", "control": ""})

@app.route('/update_position', methods=['POST'])
def update_position():
    global player_data
    if DEBUG: print('🚨 update_position >>>')
    data = request.get_json()
    if not data or "position" not in data:
        if DEBUG: print("🚫 Missing position data")
        return jsonify({"status": "ERROR", "message": "Missing position data"}), 400

    try:
        x, y, z = map(float, data["position"].split(","))
        player_data['pos'] = {'x': x, 'y': y, 'z': z}
        player_data.setdefault('turret_x', 0)
        player_data.setdefault('turret_y', 0)
        player_data.setdefault('body_x', 0)
        player_data.setdefault('body_y', 0)
        player_data.setdefault('body_z', 0)
        if DEBUG: print(f"📍 Position updated: {player_data['pos']}")
        return jsonify({"status": "OK", "current_position": player_data['pos']})
    except Exception as e:
        if DEBUG: print(f"🚫 Invalid position format: {str(e)}")
        return jsonify({"status": "ERROR", "message": str(e)}), 400

@app.route('/get_move', methods=['GET'])
def get_move():
    if DEBUG: print('🚨 get_move >>>')
    global move_command
    if move_command:
        command = move_command.pop(0)
        if DEBUG: print(f"🚗 Move Command: {command}")
        return jsonify(command)
    else:
        return jsonify({"move": "STOP", "weight": 1.0})

@app.route('/get_action', methods=['GET'])
def get_action():
    global action_command, latest_nearest_enemy, first_action_state, hit_state
    if DEBUG: print('🚨 get_action >>>', action_command)
    if action_command:
        command = action_command.pop(0)
        if DEBUG: print(f"🔫 Action Command: {command}")
        
        if hit_state == 1 and command['turret'] != 'FIRE' and command['weight'] == 0.1:
            # reverse 끝나는 지점
            first_action_state = True
            hit_state = -1
            # print("impact_control False", action_command)
            if STATE_DEBUG : print('5 🤩🤩reverse end - first_action_state t', first_action_state)
            if STATE_DEBUG : print('5 🤩🤩reverse end - hit_state -1', hit_state)

        return jsonify(command)
    else:
        return jsonify({"turret": "", "weight": 0.0})

@app.route('/update_bullet', methods=['POST'])
def update_bullet():
    global destination, impact_info, player_data, action_command, latest_nearest_enemy, hit_state
    if DEBUG: print('🚨 update_bullet >>>')
    data = request.get_json()
    print("😍😍",data)
    action_command = []
    if not data:
        if DEBUG: print("🚫 Invalid bullet data")
        return jsonify({"status": "ERROR", "message": "Invalid request data"}), 400

    print(f"💥 Bullet Impact at X={data.get('x')}, Y={data.get('y')}, Z={data.get('z')}, Target={data.get('hit')}")
    impact_info = {
        'x': data.get('x'),
        'y': data.get('y'),
        'z': data.get('z'),
        'target': data.get('hit'),
        'timestamp': time.strftime('%H:%M:%S')
    }

    # 수정필요: 타겟 명중 여부 판단 tolerence -> class_name
    is_hit = turret.is_hit(latest_nearest_enemy, impact_info)
    if DEBUG: print('💥', is_hit)
    # 명중 못했을 때 
    if not is_hit:
        time.sleep(5)
        hit_state = 0
        try:
            action_command = turret.get_action_command(
                player_data.get('pos', {'x': 60, 'y': 10, 'z': 57}),
                latest_nearest_enemy,
                turret_x_angle=player_data.get('turret_x', 0),
                turret_y_angle=player_data.get('turret_y', 0),
                player_y_angle=player_data.get('body_y', 0)
            )
            if DEBUG: print('is_hit >> action_command:', action_command)
        except ValueError as e:
            if DEBUG: print(f"🚫 Error generating action command: {str(e)}")
            action_command = []
        
        if STATE_DEBUG : print('3 🤩🤩re action - first_action_state f', first_action_state)
        if STATE_DEBUG : print('3 🤩🤩re action - hit_state 0', hit_state)
    else:
        if DEBUG: print("Hit!!!!!")
        hit_state = 1
        if is_hit:
            print("🎯 Target HIT confirmed.")
            # 💀 dead_list에 등록
            dead_list.append({
                "x" : impact_info['x'],
                "z": impact_info['z']
            })
            print(dead_list)
        action_command = turret.get_reverse_action_command(
            player_data.get('turret_x', 0),
            player_data.get('turret_y', 0),
            player_data.get('body_x', 0),
            player_data.get('body_y', 0),
        )
        
        if STATE_DEBUG : print('4 🤩🤩reverse - first_action_state f', first_action_state)
        if STATE_DEBUG : print('4 🤩🤩reverse - hit_state 1', hit_state)

    if DEBUG: print(f"💥 Bullet Impact at X={impact_info['x']}, Y={impact_info['y']}, Z={impact_info['z']}, Target={impact_info['target']}")

    socketio.emit('bullet_impact', impact_info)
    return jsonify({"status": "OK", "message": "Bullet impact data received"})

@app.route('/set_destination', methods=['POST'])
def set_destination():
    global destination, action_command
    if DEBUG: print('🚨 set_destination >>>')
    data = request.get_json()
    action_command = []
    if not data or "destination" not in data:
        if DEBUG: print("🚫 Missing destination data")
        return jsonify({"status": "ERROR", "message": "Missing destination data"}), 400

    try:
        x, y, z = map(float, data["destination"].split(","))
        destination = {'x': x, 'y': y, 'z': z}
        if DEBUG: print(f"🎯 Destination set to: {destination}")
        action_command = turret.get_action_command(
            player_data.get('pos', {'x': 60, 'y': 10, 'z': 57}),
            destination,
            turret_x_angle=player_data.get('turret_x', 0),
            turret_y_angle=player_data.get('turret_y', 0),
            player_y_angle=player_data.get('body_y', 0)
        )
        if DEBUG: print('action_command:', action_command)
        return jsonify({"status": "OK", "destination": destination})
    except Exception as e:
        if DEBUG: print(f"🚫 Invalid destination format: {str(e)}")
        return jsonify({"status": "ERROR", "message": f"Invalid format: {str(e)}"}), 400

def get_center(obstacle_info):
    xc = (obstacle_info['x_min'] + obstacle_info['x_max'])/2
    zc = (obstacle_info['z_min'] + obstacle_info['z_max'])/2
    return xc, zc

@app.route('/update_obstacle', methods=['POST'])
def update_obstacle():
    global obstacles
    if DEBUG: print('🚨 update_obstacle >>>')
    data = request.get_json()
    if not data or 'obstacles' not in data:
        if DEBUG: print("🚫 No obstacle data received")
        logging.warning("No obstacle data received")
        return jsonify({'status': 'error', 'message': 'No data received'}), 400

    obstacles = data['obstacles']
    center = [get_center(x) for x in obstacles]
    print(f"🪨 Obstacle data updated: {len(obstacles)} items")
    logging.debug(f"Obstacle data updated: {json.dumps(obstacles, indent=2)}")
    if DEBUG: print(f"🪨 Obstacle data: {json.dumps(obstacles, indent=2)}")
    return jsonify({'status': 'success', 'message': 'Obstacle data received', 'obstacles_count': len(obstacles)})

@app.route('/init', methods=['GET'])
def init():
    if DEBUG: print('🚨 init >>>')

    config = {
        "startMode": "start",
        "blStartX": 60,
        "blStartY": 10,
        "blStartZ": 57,
        "rdStartX": 60,
        "rdStartY": 10,
        "rdStartZ": 280,
        "detectMode": False,
        "trackingMode": False,
        "logMode": False,
        "enemyTracking": False,
        "saveSnapshot": False,
        "saveLog": False,
        "saveLidarData": False,
        "lux": 30000
    }

    if DEBUG: print(f"🛠️ Initialization config sent via /init: {config}")
    return jsonify(config)

@app.route('/start', methods=['GET'])
def start():
    if DEBUG: print("🚀 /start command received")
    return jsonify({"control": ""})

def wait_for_impact_confirm(timeout=3.0):
    """/update_bullet로 명중 여부가 반영될 때까지 기다림"""
    global hit_state
    start_time = time.time()
    print("⏳ 포격 후 명중 여부 확인 중...")

    while time.time() - start_time < timeout:
        if hit_state in [0, 1]:  # 0=miss, 1=hit
            print(f"✅ 명중 여부 확인 완료: hit_state={hit_state}")
            return
        time.sleep(0.1)  # 100ms 단위로 확인

    print("⚠️ 제한 시간 내 명중 여부 확인 실패")
    
@app.route('/start_rotation', methods=['POST'])
def start_rotation():
    global action_command,  player_data, obstacles, dead_list, latest_nearest_enemy
    print('🚨 start_rotation >>>')
    if DEBUG: print('🚨 start_rotation >>>')

    for _ in range(36):  # 360도 회전 (10도씩)
        # 1. 회전 명령 큐에 추가 (Q: 좌회전)
        action_command.append({"turret": "Q", "weight": 0.1})
        action_command.append({"turret": "Q", "weight": 0.0})  # 회전 멈춤

        # 2. YOLO 탐지 요청 (클라이언트에서 이미지 주기적으로 보내줘야 함)
        # 서버 내 이미지 경로로 예시 처리
        image_path = 'temp_image.jpg'
        try:
            results = model(image_path, imgsz=640)
            detections = results[0].boxes.data.cpu().numpy()
        except:
            continue  # 예외 발생 시 다음 회전으로

        # 3. 탐지 결과 처리
        filtered_results = []
        target_classes = {0: 'car002', 1: 'tank'}
        for box in detections:
            class_id = int(box[5])
            if class_id not in target_classes:
                continue
            bbox = [float(coord) for coord in box[:4]]
            filtered_results.append({
                'className': target_classes[class_id],
                'bbox': bbox,
                'confidence': float(box[4])
            })

        # 4. 가장 가까운 적 탐색
        nearest_enemy = get_enemy_pos.find_nearest_enemy(filtered_results, player_data, obstacles)
        if not nearest_enemy['state']:
            continue  # 탐지된 적 없음
        ex = nearest_enemy['x']
        ez = nearest_enemy['z']
        if get_enemy_pos.is_already_dead(ex, ez, dead_list):
            print("🧟‍♂️ 이미 사망한 타겟. 포격 제외.")
            continue

        # 5. 포격 명령 추가
        latest_nearest_enemy = nearest_enemy
        firing_cmds = turret.get_action_command(
            'auto',
            player_data['pos'],
            nearest_enemy,
            turret_x_angle=player_data.get('turret_x', 0),
            turret_y_angle=player_data.get('turret_y', 0),
            player_y_angle=player_data.get('body_y', 0)
        )
        action_command += firing_cmds

        wait_for_impact_confirm(timeout=3.0)

        print("🎯 타겟 포착 및 포격 명령 추가됨:", nearest_enemy)
        break  # 하나만 포착하고 종료하려면 break / 전체 순회하려면 제거

    return jsonify({"status": "OK", "message": "Auto rotation and targeting initiated."})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)


# # 4. YOLO 탐지
# results = model(image_path, imgsz=640)
# detections = results[0].boxes.data.cpu().numpy()
# logging.debug(f"YOLO detections: {len(detections)} objects")

# # 5. 탐지 결과 필터링
# target_classes = {
#     0: 'car002', 1: 'car003', 2: 'car005', 3: 'human001',
#     4: 'rock001', 5: 'rock2', 6: 'tank', 7: 'wall001', 8: 'wall002'
# }
# class_colors = {
#     'car002': 'red', 'car003': 'blue', 'car005': 'green', 'human001': 'orange',
#     'rock001': 'purple', 'rock2': 'yellow', 'tank': 'cyan', 'wall001': 'pink', 'wall002': 'brown'
# }
# try:
#     font = ImageFont.truetype("arial.ttf", size=20)
# except:
#     font = ImageFont.load_default()

# filtered_results = []

# img = Image.open(image_path).convert('RGB')
# draw = ImageDraw.Draw(img)

# for row in detections:
#     class_id = int(row[5])
#     if class_id not in target_classes:
#         continue
# target_name = target_classes[class_id]
# bbox_yolo = [float(item) for item in row[:4]]
# confidence = float(row[4])
# bbox = [
#     bbox_yolo[0] * 1920 / 640,
#     bbox_yolo[1] * 1080 / 640,
#     bbox_yolo[2] * 1920 / 640,
#     bbox_yolo[3] * 1080 / 640
# ]
# coords = calculate_map_coords(
#     fov_horizontal, fov_vertical, player_pose,
#     width, height, map_width, map_height,
#     bbox
# )
# filtered_results.append({
#     'className': target_name,
#     'bbox': bbox,
#     'confidence': confidence,
#     'map_center': coords['map_center']
# })
# for box in detections:
#     class_id = int(box[5])
#     if class_id in target_classes: 
#         filtered_results.append({
#             'className': target_classes[class_id],
#             'bbox': [float(coord) for coord in box[:4]],
#             'confidence': float(box[4])
#         })

# # 6. obstacles와 detections 매칭
# matched_obstacles = is_near_enemy.match_obstacles_with_detections(obstacles, filtered_results, threshold=MATCH_THRESHOLD)

# # 7. 가장 가까운 적 찾기
# nearest_enemy = is_near_enemy.find_nearest_enemy(filtered_results, player_pose, matched_obstacles, match_threshold=MATCH_THRESHOLD)
# fire_coordinates = is_near_enemy.get_fire_coordinates(nearest_enemy)
# latest_nearest_enemy = nearest_enemy
# print(f"🐟 nearest_enemy: {nearest_enemy}")
# print(f"🪡 fire_coordinates: {fire_coordinates}")
# logging.debug(f"Nearest enemy: {nearest_enemy}")
# logging.debug(f"Fire coordinates: {fire_coordinates}")


# result_list = []
# for row in detections: 
#     target_name = target_classes[row[-1]]
#     target_bbox = [float(item) for item in row[:4]]
#     confidence = float(row[-2])
#     result_list.append({'className': target_name, 'bbox': target_bbox, 'confidence': confidence})


# # 플레이어 위치
# player_pos = (
#     player_data.get('pos', {}).get('x', 60),
#     player_data.get('pos', {}).get('z', 57)
# )
# print("Player position:", player_pos)

# # 8. 포격 좌표 설정 및 액션 커맨드
# if 'message' not in fire_coordinates:
#     destination = {'x': fire_coordinates['x'], 'y': 10, 'z': fire_coordinates['z']}
#     try:
#         if DEBUG: print(f"👉 Generating action command: player_pose={player_pose}, dest={destination}")
#         action_command = turret.get_action_command(
#             player_pose,
#             destination,
#             turret_x_angle=player_data.get('turret_x', 0),
#             turret_y_angle=player_data.get('turret_y', 0),
#             player_y_angle=player_data.get('body_y', 0)
#         )
#         print(f"🎯 Auto-set destination: {destination}")
#     except ValueError as e:
#         print(f"🚫 Error generating action command: {str(e)}")
#         action_command = []

# # 9. 전체 이미지의 맵 비율
# image_coords = calculate_map_coords(
#     fov_horizontal, fov_vertical, player_pose,
#     width, height, map_width, map_height,
#     [0, 0, width, height]
# )

# # 10. 로그 및 응답
# if 'message' in nearest_enemy:
#     enemy_log = f"Nearest enemy: {nearest_enemy['message']}"
# else:
#     enemy_log = (
#         f"Nearest enemy: x={nearest_enemy['x']:.6f}, z={nearest_enemy['z']:.6f}, "
#         f"class={nearest_enemy['className']}, confidence={nearest_enemy['confidence']:.2f}, "
#         f"source={nearest_enemy['source']}"
#     )
# print(f"🚀 {enemy_log}")
# print(f"🎯 Fire coordinates: {fire_coordinates}")

# response = {
#     'detections': filtered_results,
#     'matched_obstacles': matched_obstacles,
#     'nearest_enemy': nearest_enemy,
#     'fire_coordinates': fire_coordinates,
#     'image_real_size': image_coords['image_real_size'],
#     'map_ratio': image_coords['map_ratio'],
#     'fov': {'horizontal': fov_horizontal, 'vertical': fov_vertical},
#     'match_threshold': MATCH_THRESHOLD,
#     'control': 'continue'
# }
# if DEBUG: print(f"Detection response: {json.dumps(response, indent=2)}")
# return jsonify(response)