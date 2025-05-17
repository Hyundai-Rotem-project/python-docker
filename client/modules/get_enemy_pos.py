import math
import logging

#3 FOV 및 카메라 설정
FOV_HORIZONTAL = 50
FOV_VERTICAL = 28
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
MAP_WIDTH = 300
MAP_HEIGHT = 300

# 로깅 설정
logging.basicConfig(filename='tank.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_relative_angle(player_data, obstacle_info):
    player_pos = player_data['pos']
    player_facing = {
        'x': player_data['body_x'],
        'y': player_data['body_y'],
        'z': player_data['body_z']
    }

    for index, obs in enumerate(obstacle_info):
        position = obs['position']

        # 벡터: player → obstacle
        dx = position['x'] - player_pos['x']
        dz = position['z'] - player_pos['z']
        
        # player의 바라보는 방향 (기준 벡터)
        facing_angle = round((player_facing['x'] + 180) % 360 - 180, 2)
        obstacle_angle = round(math.degrees(math.atan2(dx, dz)), 2)

        relative_angle = obstacle_angle - facing_angle

        target = obstacle_info[index]
        target['angle'] = relative_angle
        
    return obstacle_info

def calculate_bbox_angle(bbox):
    bbox_cx = (bbox[0] + bbox[2])/2
    dx = (bbox_cx - IMAGE_WIDTH / 2) / (IMAGE_WIDTH / 2)
    angle_x = dx * (FOV_HORIZONTAL / 2)
    return angle_x

def match_bbox_to_obstacle(detected_results, player_data, obstacle_data):
    obstacle_info = calculate_relative_angle(player_data, obstacle_data)
    obstacle_angle = [item['angle'] for item in obstacle_info]
        
    for index, det in enumerate(detected_results):
        bbox = det['bbox']

        bbox_angle = calculate_bbox_angle(bbox)
        min_angle = float('inf')
        for i, obs in enumerate(obstacle_info):
            angel_diff = abs(obs['angle'] - bbox_angle)
            cond = angel_diff < min_angle and det['className'] == obs['prefabName']
            if cond:
                min_angle = angel_diff
                det['position'] = obs['position']
                det['id'] = obs['id']

        if det.get('position') is None:
            det['className'] = 'Miss_detected'
            
    return detected_results

def get_enemy_list(detections, player_data, obstacles):
    """가장 가까운 적 반환 (1200m 이내 적만 valid_enemies로 간주)"""
    detected_results = match_bbox_to_obstacle(detections, player_data, obstacles)
    player_pos = player_data['pos']
    logging.debug(f"Starting find_nearest_enemy with {len(detections)} detections, player_pos: {player_pos}, obstacles: {len(obstacles)}")
    
    enemy_classes = {'Car002', 'Tank001'}  # 적 클래스
    detected_classes = {det['className'] for det in detections if det['className'] in enemy_classes and det['confidence'] >= 0.3}
    logging.debug(f"Detected classes: {detected_classes}")
    
    if not detected_classes:
        logging.info("No enemy classes detected")
        return {'message': 'No enemy detected', 'state': False}
    
    if not player_pos:
        logging.warning("Player position not set")
        return {'message': 'Player position not set', 'state': False}

    valid_enemies = []
    for detected in detected_results:
        print('detected', detected)
        if detected['className'] in enemy_classes and detected['confidence'] > 0.3:
            center_x = detected['position']['x']
            center_y = detected['position']['y']
            center_z = detected['position']['z']
            # 플레이어와의 거리 계산
            distance = math.sqrt((center_x - player_pos['x'])**2 + (center_z - player_pos['z'])**2)
            if distance <= 1200:  # 1200m 이내인 경우만 추가
                valid_enemies.append({
                    'x': center_x,
                    'z': center_z,
                    'y': center_y,
                    'className': detected['className'],
                    # 'confidence': 1.0,  # /set_obstacles 데이터 신뢰도
                    # 'source': 'obstacles',
                    'distance': distance,
                    'id': detected['id']
                })
                logging.debug(f"Valid enemy added: x={center_x:.2f}, z={center_z:.2f}, distance={distance:.2f}m")
            else:
                logging.debug(f"Enemy excluded (too far): x={center_x:.2f}, z={center_z:.2f}, distance={distance:.2f}m")

    if not valid_enemies:
        logging.info("No matching enemies within 1200m")
        return {'message': 'No matching enemy found within 1200m', 'state': False}
    
    print('valid_enemies', valid_enemies)
    sorted_valid_enemies = sorted(valid_enemies, key=lambda x: x['distance'])
    enemy_list = {'list': sorted_valid_enemies, 'state': True}

    return enemy_list

def find_nearest_enemy(detections, player_data, obstacles):
    """가장 가까운 적 반환 (1200m 이내 적만 valid_enemies로 간주)"""
    detected_results = match_bbox_to_obstacle(detections, player_data, obstacles)
    player_pos = player_data['pos']
    logging.debug(f"Starting find_nearest_enemy with {len(detections)} detections, player_pos: {player_pos}, obstacles: {len(obstacles)}")
    
    enemy_classes = {'Car002', 'Tank001'}  # 적 클래스
    detected_classes = {det['className'] for det in detections if det['className'] in enemy_classes and det['confidence'] >= 0.3}
    logging.debug(f"Detected classes: {detected_classes}")
    
    if not detected_classes:
        logging.info("No enemy classes detected")
        return {'message': 'No enemy detected', 'state': False}
    
    if not player_pos:
        logging.warning("Player position not set")
        return {'message': 'Player position not set', 'state': False}

    valid_enemies = []
    for detected in detected_results:
        print('detected', detected)
        if detected['className'] in enemy_classes and detected['confidence'] > 0.3:
            center_x = detected['position']['x']
            center_y = detected['position']['y']
            center_z = detected['position']['z']
            # 플레이어와의 거리 계산
            distance = math.sqrt((center_x - player_pos['x'])**2 + (center_z - player_pos['z'])**2)
            if distance <= 1200:  # 1200m 이내인 경우만 추가
                valid_enemies.append({
                    'x': center_x,
                    'z': center_z,
                    'y': center_y,
                    'className': detected['className'],
                    # 'confidence': 1.0,  # /set_obstacles 데이터 신뢰도
                    # 'source': 'obstacles',
                    'distance': distance,
                    'id': detected['id']
                })
                logging.debug(f"Valid enemy added: x={center_x:.2f}, z={center_z:.2f}, distance={distance:.2f}m")
            else:
                logging.debug(f"Enemy excluded (too far): x={center_x:.2f}, z={center_z:.2f}, distance={distance:.2f}m")

    if not valid_enemies:
        logging.info("No matching enemies within 1200m")
        return {'message': 'No matching enemy found within 1200m', 'state': False}
    
    print('valid_enemies', valid_enemies)
    sorted_valid_enemies = sorted(valid_enemies, key=lambda x: x['distance'])
    min_distance = float('inf')
    nearest_enemy = None
    for enemy in valid_enemies:
        # print('😡valid_enemies', enemy)
        logging.info("😡valid_enemies")
        if enemy['distance'] < min_distance:
            min_distance = enemy['distance']
            nearest_enemy = enemy
            nearest_enemy['state'] = True
            # nearest_enemy = {
            #     'x': enemy['x'],
            #     'z': enemy['z'],
            #     'y': 10.0,
            #     # 'distance': enemy['distance'],
            #     # # 'className': enemy['className'],
            #     # 'confidence': enemy['confidence'],
            #     # 'source': enemy['source']
            # }
    
    if nearest_enemy:
        logging.debug(f"Nearest enemy: {nearest_enemy}")
    else:
        logging.info("No nearest enemy found after filtering")
        return {'message': 'No valid enemy found within 1200m', 'state': False}
    
    return nearest_enemy