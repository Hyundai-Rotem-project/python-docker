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
    # 객체와 전차와의 상대적 각도 계산
    player_pos = player_data['pos']
    player_facing = {
        'x': player_data['body_x'],
        'y': player_data['body_y'],
        'z': player_data['body_z']
    }

    for index, obs in enumerate(obstacle_info):
        xc = (obs['x_min'] + obs['x_max'])/2
        zc = (obs['z_min'] + obs['z_max'])/2

        # 벡터: player → obstacle
        dx = xc - player_pos['x']
        dz = zc - player_pos['z']
        
        # player의 바라보는 방향 (기준 벡터)
        facing_angle = round((player_facing['x'] + 180) % 360 - 180, 2)
        obstacle_angle = round(math.degrees(math.atan2(dx, dz)), 2)

        relative_angle = obstacle_angle + facing_angle

        target = obstacle_info[index]
        target['angle'] = relative_angle
        target['center'] = (xc, zc)
        
    return obstacle_info

def calculate_bbox_angle(bbox, turret_y):
    bbox_cx = (bbox[0] + bbox[2])/2

    dx = (bbox_cx - IMAGE_WIDTH / 2) / (IMAGE_WIDTH / 2)
    
    #시야각 비율만큼 회전된 각도 계산
    local_angle = dx * (FOV_HORIZONTAL / 2)
    
    #turret의 회전각을 기준으로 보정
    global_angle = (turret_y + local_angle + 360) % 360

    return global_angle

def match_bbox_to_obstacle(detected_results, player_data, obstacle_data):
    # print('🎶🤢detected_results', detected_results)
    # print('🎶🤢player_data', player_data)
    # print('obstacle_info', obstacle_info)
    
    obstacle_info = calculate_relative_angle(player_data, obstacle_data)
    obstacle_angle = [item['angle'] for item in obstacle_info]
    
    if not obstacle_info:
        print("⚠️ No obstacles to match with detections.")
        return detected_results  # 그대로 반환 (center 없음)
    
    for index, det in enumerate(detected_results):
        bbox = det['bbox']

        bbox_angle = calculate_bbox_angle(bbox)
        # 🚨 closest_index 동일하게 나오는 경우 있음 -> 각 detected_results가 다른 index 갖도록 수정 필요
        if not obstacle_angle:
            continue

        closest_index = min(range(len(obstacle_angle)), key=lambda i: abs(obstacle_angle[i] - bbox_angle))
        # print('bboxxxxx', bbox_angle, closest_index)
        obs = obstacle_info[closest_index]
        det['center'] = obs['center']

    # print("detected_results !!!", detected_results)
    return detected_results

# 🎶🤢detected_results 예시
# {'id': 0, 'className': 'car003', 
#  'bbox': [729.469482421875, 545.0945434570312, 865.4412231445312, 665.6178588867188], 
#  'confidence': 0.8793855905532837, 'color': '#0000FF', 
# 'filled': False, 'updateBoxWhileMoving': False, 
# 'center': (56.05139923095703, 99.60943603515625)}


def find_nearest_enemy(detections, player_data, obstacles):
    """가장 가까운 적 반환 (1200m 이내 적만 valid_enemies로 간주)"""
    detected_results = match_bbox_to_obstacle(detections, player_data, obstacles)
    # print('🎶🤢 player_data', player_data)
    # nearest_enemy = find_nearest_enemy(filtered_results, player_data['pos'], obstacles)
    player_pos = player_data['pos']
    logging.debug(f"Starting find_nearest_enemy with {len(detections)} detections, player_pos: {player_pos}, obstacles: {len(obstacles)}")
    
    enemy_classes = {'tank'}  # 적 클래스
    detected_classes = {det['className'] for det in detections if det['className'] in enemy_classes and det['confidence'] >= 0.3}
    # print('😡???', detected_classes)
    logging.debug(f"Detected classes: {detected_classes}")
    
    if not detected_classes:
        logging.info("No enemy classes detected")
        return {'message': 'No enemy detected', 'state': False}
    
    if not player_pos:
        logging.warning("Player position not set")
        return {'message': 'Player position not set', 'state': False}

    valid_enemies = []
    print("detected_results")
    for detected in detected_results:
        # print('detected', detected)
        if detected['className'] in enemy_classes and detected['confidence'] > 0.3:
            center_x = detected['center'][0]
            center_z = detected['center'][1]
            # 플레이어와의 거리 계산
            distance = math.sqrt((center_x - player_pos['x'])**2 + (center_z - player_pos['z'])**2)
            if distance <= 1200:  # 1200m 이내인 경우만 추가
                valid_enemies.append({
                    'x': center_x,
                    'z': center_z,
                    'y': 8,
                    'className': detected['className'],
                    # 'confidence': 1.0,  # /set_obstacles 데이터 신뢰도
                    # 'source': 'obstacles',
                    'distance': distance
                })
                logging.debug(f"Valid enemy added: x={center_x:.2f}, z={center_z:.2f}, distance={distance:.2f}m")
            else:
                logging.debug(f"Enemy excluded (too far): x={center_x:.2f}, z={center_z:.2f}, distance={distance:.2f}m")

    if not valid_enemies:
        logging.info("No matching enemies within 1200m")
        return {'message': 'No matching enemy found within 1200m', 'state': False}
    
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
    
    # print('nearest_enemy', nearest_enemy)
    return nearest_enemy

# modules/get_enemy_pos.py 또는 is_near_enemy.py 안에 추가
def is_already_dead(x, z, dead_list, threshold=2.0):
    for dead in dead_list:
        dx = x - dead['x']
        dz = z - dead['z']
        if math.sqrt(dx**2 + dz**2) <= threshold:
            return True
    return False
