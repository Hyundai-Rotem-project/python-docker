from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO
from PIL import Image, ImageDraw, ImageFont
import torch
from ultralytics import YOLO
import time
import logging
import json
import modules.turret as turret
import modules.is_near_enemy as is_near_enemy

app = Flask(__name__)

# 로깅 설정
logging.basicConfig(filename='tank.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# YOLO 모델 로드
try:
    model = YOLO('best.pt')
    logging.info("YOLO model loaded successfully")
except Exception as e:
    logging.error(f"Failed to load YOLO model: {str(e)}")
    raise RuntimeError(f"YOLO model loading failed: {str(e)}")

socketio = SocketIO(app)

# 전역 변수
move_command = []
action_command = []
player_data = {'pos': {'x': 60, 'y': 10, 'z': 57}}  # 기본 위치 설정
destination = {}
impact_info = {}
obstacles = []  # /set_obstacles 데이터 저장
latest_nearest_enemy = None

@app.route('/dashboard')
def dashboard():
    print('🚨 dashboard >>>')
    logging.info("Dashboard accessed")
    return render_template('dashboard.html')

@app.route('/detect', methods=['POST'])
def detect():
    global player_data, obstacles, latest_nearest_enemy, action_command, destination
    print('🚨 detect >>>')
    logging.debug("Receiving /detect request")

    # 이미지 수신
    image = request.files.get('image')
    if not image:
        logging.error("No image received in /detect")
        return jsonify({"error": "No image received", "control": "continue"}), 400

    image_path = 'temp_image.jpg'
    try:
        image.save(image_path)
        logging.info(f"Image saved to {image_path}")
    except Exception as e:
        logging.error(f"Failed to save image: {str(e)}")
        return jsonify({"error": f"Failed to save image", "control": "continue"}), 500

    # YOLO 탐지
    try:
        results = model(image_path, imgsz=640)
        detections = results[0].boxes.data.cpu().numpy()
        logging.debug(f"YOLO raw detections: {detections.tolist()}")
    except Exception as e:
        logging.error(f"YOLO detection failed: {str(e)}")
        return jsonify({"error": f"YOLO detection failed: {str(e)}", "control": "continue"}), 500

    # 탐지 결과 필터링
    target_classes = {
        0: 'car002', 1: 'car003', 2: 'car005', 3: 'human001',
        4: 'rock001', 5: 'rock2', 6: 'tank', 7: 'wall001', 8: 'wall002'
    }
    class_colors = {
        'car002': 'red', 'car003': 'blue', 'car005': 'green', 'human001': 'orange',
        'rock001': 'purple', 'rock2': 'yellow', 'tank': 'cyan', 'wall001': 'pink', 'wall002': 'brown'
    }

    filtered_results = []
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", size=20)
    except:
        font = ImageFont.load_default()

    for box in detections:
        confidence = float(box[4])
        class_id = int(box[5])
        if confidence >= 0.5 and class_id in target_classes:  # 신뢰도 0.5로 낮춤
            class_name = target_classes[class_id]
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            width = x2 - x1
            height = y2 - y1

            if class_name == 'wall002' and width * height < 5000:
                print(f"🚫 작은 wall002 무시: width={width}, height={height}")
                continue

            color = class_colors.get(class_name, 'white')
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.text((x1, y1 - 20), f"{class_name} {confidence:.2f}", fill=color, font=font)

            filtered_results.append({
                'className': class_name,
                'bbox': [float(coord) for coord in box[:4]],
                'confidence': confidence
            })
            print("Filtered result:", filtered_results[-1])

    # 플레이어 위치
    player_pos = (
        player_data.get('pos', {}).get('x', 60),
        player_data.get('pos', {}).get('z', 57)
    )
    print("Player position:", player_pos)
    logging.debug(f"Player position: {player_pos}")

    # 가장 가까운 적 찾기
    nearest_enemy = is_near_enemy.find_nearest_enemy(filtered_results, player_pos, obstacles)
    fire_coordinates = is_near_enemy.get_fire_coordinates(nearest_enemy)
    latest_nearest_enemy = nearest_enemy

    # 포격 좌표 설정
    if 'message' not in fire_coordinates:
        destination = {'x': fire_coordinates['x'], 'y': 10, 'z': fire_coordinates['z']}
        try:
            print(f"👉 Generating action command: player_pos={player_data.get('pos')}, dest={destination}")
            action_command = turret.get_action_command(
                player_data.get('pos', {'x': 60, 'y': 10, 'z': 57}),
                destination,
                turret_x_angle=player_data.get('turret_x', 0),
                turret_y_angle=player_data.get('turret_y', 0),
                player_y_angle=player_data.get('body_y', 0)
            )
            logging.info(f"Auto-set destination: {destination}")
            print(f"🎯 Auto-set destination: {destination}")
        except ValueError as e:
            logging.error(f"Error generating action command: {str(e)}")
            print(f"🚫 Error generating action command: {str(e)}")
            action_command = []

    # 로그 및 응답
    if 'message' in nearest_enemy:
        enemy_log = f"Nearest enemy: {nearest_enemy['message']}"
    else:
        enemy_log = (
            f"Nearest enemy: x={nearest_enemy['x']:.6f}, z={nearest_enemy['z']:.6f}, "
            f"class={nearest_enemy['className']}, confidence={nearest_enemy['confidence']:.2f}, "
            f"source={nearest_enemy['source']}"
        )
    logging.info(enemy_log)
    print(f"🚀 {enemy_log}")

    fire_log = f"Fire coordinates: {fire_coordinates}"
    logging.info(fire_log)
    print(f"🎯 {fire_log}")

    response = {
        'detections': filtered_results,
        'nearest_enemy': nearest_enemy,
        'fire_coordinates': fire_coordinates,
        'control': 'continue'
    }
    logging.debug(f"Detection response: {json.dumps(response, indent=2)}")
    print(f"Detection response: {json.dumps(response, indent=2)}")
    return jsonify(response)

@app.route('/info', methods=['POST'])
def info():
    global player_data
    logging.debug("Receiving /info request")
    data = request.get_json(force=True)
    if not data:
        logging.error("No JSON received")
        print("🚫 No JSON received")
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
    logging.debug(f"Player data updated: {player_data}")
    print(f"📍 Player data updated: {player_data}")
    return jsonify({"status": "success", "control": ""})

@app.route('/update_position', methods=['POST'])
def update_position():
    global player_data
    print('🚨 update_position >>>')
    logging.debug("Receiving /update_position request")
    data = request.get_json()
    if not data or "position" not in data:
        logging.error("Missing position data")
        print("🚫 Missing position data")
        return jsonify({"status": "ERROR", "message": "Missing position data"}), 400

    try:
        x, y, z = map(float, data["position"].split(","))
        player_data['pos'] = {'x': x, 'y': y, 'z': z}
        player_data.setdefault('turret_x', 0)
        player_data.setdefault('turret_y', 0)
        player_data.setdefault('body_x', 0)
        player_data.setdefault('body_y', 0)
        player_data.setdefault('body_z', 0)
        logging.info(f"Position updated: {player_data['pos']}")
        print(f"📍 Position updated: {player_data['pos']}")
        return jsonify({"status": "OK", "current_position": player_data['pos']})
    except Exception as e:
        logging.error(f"Invalid position format: {str(e)}")
        print(f"🚫 Invalid position format: {str(e)}")
        return jsonify({"status": "ERROR", "message": str(e)}), 400

@app.route('/get_move', methods=['GET'])
def get_move():
    print('🚨 get_move >>>')
    global move_command
    logging.debug("Receiving /get_move request")
    if move_command:
        command = move_command.pop(0)
        logging.info(f"Move Command: {command}")
        print(f"🚗 Move Command: {command}")
        return jsonify(command)
    else:
        return jsonify({"move": "STOP", "weight": 1.0})

@app.route('/get_action', methods=['GET'])
def get_action():
    global action_command, latest_nearest_enemy
    print('🚨 get_action >>>', action_command)
    logging.debug("Receiving /get_action request")

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
            logging.error(f"Error generating action command: {str(e)}")
            print(f"🚫 Error generating action command: {str(e)}")
            action_command = []

    if action_command:
        command = action_command.pop(0)
        logging.info(f"Action Command: {command}")
        print(f"🔫 Action Command: {command}")
        return jsonify(command)
    else:
        return jsonify({"turret": "", "weight": 0.0})

@app.route('/update_bullet', methods=['POST'])
def update_bullet():
    global destination, impact_info, player_data, action_command
    print('🚨 update_bullet >>>')
    logging.debug("Receiving /update_bullet request")
    data = request.get_json()
    action_command = []
    if not data:
        logging.error("Invalid bullet data")
        print("🚫 Invalid bullet data")
        return jsonify({"status": "ERROR", "message": "Invalid request data"}), 400

    impact_info = {
        'x': data.get('x'),
        'y': data.get('y'),
        'z': data.get('z'),
        'target': data.get('hit'),
        'timestamp': time.strftime('%H:%M:%S')
    }

    is_hit = turret.is_hit(destination, impact_info)
    logging.info(f"Bullet hit check: {is_hit}")
    print('💥', is_hit)
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
            print('is_hit >> action_command:', action_command)
        except ValueError as e:
            logging.error(f"Error generating action command: {str(e)}")
            print(f"🚫 Error generating action command: {str(e)}")
            action_command = []
    else:
        print("Hit!!!!!")
        action_command = turret.get_reverse_action_command(
            player_data.get('turret_x', 0),
            player_data.get('turret_y', 0),
            player_data.get('body_y', 0)
        )

    logging.info(f"Bullet Impact at X={impact_info['x']}, Y={impact_info['y']}, Z={impact_info['z']}, Target={impact_info['target']}")
    print(f"💥 Bullet Impact at X={impact_info['x']}, Y={impact_info['y']}, Z={impact_info['z']}, Target={impact_info['target']}")

    socketio.emit('bullet_impact', impact_info)
    return jsonify({"status": "OK", "message": "Bullet impact data received"})

@app.route('/set_destination', methods=['POST'])
def set_destination():
    global destination, action_command
    print('🚨 set_destination >>>')
    logging.debug("Receiving /set_destination request")
    data = request.get_json()
    action_command = []
    if not data or "destination" not in data:
        logging.error("Missing destination data")
        print("🚫 Missing destination data")
        return jsonify({"status": "ERROR", "message": "Missing destination data"}), 400

    try:
        x, y, z = map(float, data["destination"].split(","))
        destination = {'x': x, 'y': y, 'z': z}
        logging.info(f"Destination set to: {destination}")
        print(f"🎯 Destination set to: {destination}")
        action_command = turret.get_action_command(
            player_data.get('pos', {'x': 60, 'y': 10, 'z': 57}),
            destination,
            turret_x_angle=player_data.get('turret_x', 0),
            turret_y_angle=player_data.get('turret_y', 0),
            player_y_angle=player_data.get('body_y', 0)
        )
        print('action_command:', action_command)
        return jsonify({"status": "OK", "destination": destination})
    except Exception as e:
        logging.error(f"Invalid destination format: {str(e)}")
        print(f"🚫 Invalid destination format: {str(e)}")
        return jsonify({"status": "ERROR", "message": f"Invalid format: {str(e)}"}), 400

@app.route('/set_obstacles', methods=['POST'])
def set_obstacles():
    global obstacles
    print('🚨 set_obstacles >>>')
    logging.debug("Receiving /set_obstacles request")
    data = request.get_json()
    if not data or 'obstacles' not in data:
        logging.error("No obstacle data received")
        print("🚫 No obstacle data received")
        return jsonify({'status': 'error', 'message': 'No data received'}), 400

    obstacles = data['obstacles']
    print("Obstacle data:", obstacles)
    logging.info(f"Obstacle data updated: {json.dumps(obstacles, indent=2)}")
    print(f"🪨 Obstacle data updated: {json.dumps(obstacles, indent=2)}")
    return jsonify({'status': 'success', 'message': 'Obstacle data received'})

@app.route('/init', methods=['GET'])
def init():
    print('🚨 init >>>')
    logging.debug("Receiving /init request")
    config = {
        "startMode": "start",
        "blStartX": 60,
        "blStartY": 10,
        "blStartZ": 57,
        "rdStartX": 60,
        "rdStartY": 10,
        "rdStartZ": 280,
        "detectMode": True,
        "trackingMode": True,
        "logMode": True,
        "enemyTracking": True,
        "saveSnapshot": False,
        "saveLog": True,
        "saveLidarData": False,
        "lux": 30000
    }
    logging.info(f"Initialization config sent: {config}")
    print(f"🛠️ Initialization config sent via /init: {config}")
    return jsonify(config)

@app.route('/start', methods=['GET'])
def start():
    logging.debug("Receiving /start request")
    print("🚀 /start command received")
    return jsonify({"control": ""})

@app.route('/test_rotation', methods=['POST'])
def test_rotation():
    global action_command
    print('🚨 test_rotation >>>')
    logging.debug("Receiving /test_rotation request")
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
    logging.info(f"Testing {test_info['rotation_desc']} rotation ({rotation_type}) x {count}")
    print(f"🔄 Testing {test_info['rotation_desc']} rotation ({rotation_type}) x {count}")
    socketio.emit('rotation_test', test_info)
    print("action_command >>", action_command)
    return jsonify({"status": "OK", "message": "Rotation test started"})

if __name__ == '__main__':
    logging.info("Starting Flask server with SocketIO")
    socketio.run(app, host='0.0.0.0', port=5000)