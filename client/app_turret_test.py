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
import math

app = Flask(__name__)

DEBUG = True

# response = {
#     'detections': filtered_results,
#     'nearest_enemy': nearest_enemy,
#     'fire_coordinates': fire_coordinates,
#     'control': 'continue'
# }


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

def calculate_map_coords(fov_horizontal, fov_vertical, player_pose, image_width, image_height, map_width, map_height, bbox):
    """FOV, player_pose['y'], 이미지 픽셀 값으로 맵 비율과 사물 중심 X, Z 좌표 계산"""
    camera_height = player_pose.get('y', 8)
    width_real = 2 * camera_height * math.tan(math.radians(fov_horizontal / 2))
    height_real = 2 * camera_height * math.tan(math.radians(fov_vertical / 2))
    ratio_horizontal = (width_real / map_width) * 100
    ratio_vertical = (height_real / map_height) * 100
    x_center_pixel = (bbox[0] + bbox[2]) / 2
    y_center_pixel = (bbox[1] + bbox[3]) / 2
    meter_per_pixel_x = width_real / image_width
    meter_per_pixel_y = height_real / image_height
    x_relative_m = (x_center_pixel - image_width / 2) * meter_per_pixel_x
    z_relative_m = (y_center_pixel - image_height / 2) * meter_per_pixel_y
    x_map = player_pose.get('x', 60) + x_relative_m
    z_map = player_pose.get('z', 57) + camera_height + z_relative_m
    return {
        'image_real_size': {'width': width_real, 'height': height_real},
        'map_ratio': {'horizontal': ratio_horizontal, 'vertical': ratio_vertical},
        'map_center': {'x': x_map, 'z': z_map}
    }

@app.route('/dashboard')
def dashboard():
    if DEBUG: print('🚨 dashboard >>>')
    return render_template('dashboard.html')

@app.route('/detect', methods=['POST'])
def detect():
    global player_data, obstacles, latest_nearest_enemy, action_command, destination
    print('🌍 detect >>>')
    
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

@app.route('/info', methods=['POST'])
def info():
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
    if DEBUG: print(f"📍 Player data updated: {player_data}")
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
    global action_command, latest_nearest_enemy
    if DEBUG: print('🚨 get_action >>>', action_command)

    if latest_nearest_enemy and 'message' not in latest_nearest_enemy:
        try:
            action_command = turret.get_action_command(
                player_data.get('pos', {'x': 60, 'y': 10, 'z': 57}),
                {'x': latest_nearest_enemy['x'], 'y': 10, 'z': latest_nearest_enemy['z']},
                turret_x_angle=player_data.get('turret_x', 0),
                turret_y_angle=player_data.get('turret_y', 0),
                player_y_angle=player_data.get('body_y', 0)
            )
        except ValueError as e:
            if DEBUG: print(f"🚫 Error generating action command: {str(e)}")
            action_command = []

    if action_command:
        command = action_command.pop(0)
        if DEBUG: print(f"🔫 Action Command: {command}")
        return jsonify(command)
    else:
        return jsonify({"turret": "", "weight": 0.0})

@app.route('/update_bullet', methods=['POST'])
def update_bullet():
    global destination, impact_info, player_data, action_command
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
    # FIXME
    print("🤦‍♀️",destination, "impact_info", impact_info)
    is_hit = turret.is_hit(destination, impact_info)
    if DEBUG: print('💥', is_hit)
    if not is_hit:
        time.sleep(5)
        try:
            action_command = turret.get_action_command(
                player_data.get('pos', {'x': 60, 'y': 10, 'z': 57}),
                destination,
                turret_x_angle=player_data.get('turret_x', 0),
                turret_y_angle=player_data.get('turret_y', 0),
                player_y_angle=player_data.get('body_y', 0)
            )
            if DEBUG: print('is_hit >> action_command:', action_command)
        except ValueError as e:
            if DEBUG: print(f"🚫 Error generating action command: {str(e)}")
            action_command = []
    else:
        if DEBUG: print("Hit!!!!!")
        #FIXME
        action_command = turret.get_reverse_action_command(
            player_data.get('turret_x', 0),
            player_data.get('turret_y', 0),
            player_data.get('body_y', 0)
        )

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

@app.route('/update_obstacle', methods=['POST'])
def update_obstacle():
    global obstacles
    print('🚨 update_obstacle >>>')
    data = request.get_json()
    if not data or 'obstacles' not in data:
        if DEBUG: print("🚫 No obstacle data received")
        logging.warning("No obstacle data received")
        return jsonify({'status': 'error', 'message': 'No data received'}), 400

    obstacles = data['obstacles']
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

@app.route('/test_rotation', methods=['POST'])
def test_rotation():
    global action_command
    if DEBUG: print('🚨 test_rotation >>>')
    data = request.get_json()
    rotation_type = data.get('type', 'Q')
    count = data.get('count', 1)

    action_command = []
    for _ in range(count):
        action_command.append({"turret": rotation_type, "weight": 0.5})
    action_command.append({"turret": rotation_type, "weight": 0.0})

    test_info = {
        'rotation_type': rotation_type,
        'count': count,
        'timestamp': time.strftime('%H:%M:%S'),
        'rotation_desc': {
            'Q': 'Left', 'E': 'Right', 'F': 'Down', 'R': 'Up'
        }.get(rotation_type, 'Unknown')
    }
    if DEBUG: print(f"🔄 Testing {test_info['rotation_desc']} rotation ({rotation_type}) x {count}")
    socketio.emit('rotation_test', test_info)
    if DEBUG: print("action_command >>", action_command)
    return jsonify({"status": "OK", "message": "Rotation test started"})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)