import math
import logging

# 로깅 설정
logging.basicConfig(filename='tank.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def find_nearest_enemy(detections, player_pos, obstacles):
    """가장 가까운 적 반환 (1200m 이내 적만 valid_enemies로 간주)"""
    print('🐟', detections)
    # print('🐟', player_pos)
    # print('🐟', obstacles)
    logging.debug(f"Starting find_nearest_enemy with {len(detections)} detections, player_pos: {player_pos}, obstacles: {len(obstacles)}")
    
    enemy_classes = {'car002', 'tank'}  # 적 클래스
    # enemy_classes = {'car002', 'car003', 'tank'}  # 적 클래스
    detected_classes = {det['className'] for det in detections if det['className'] in enemy_classes and det['confidence'] >= 0.3}
    print('😡???', detected_classes)
    logging.debug(f"Detected classes: {detected_classes}")
    
    if not detected_classes:
        logging.info("No enemy classes detected")
        return {'message': 'No enemy detected', 'state': False}
    
    if not player_pos:
        logging.warning("Player position not set")
        return {'message': 'Player position not set', 'state': False}
    
    valid_enemies = []
    for obs in obstacles:
        center_x = (obs['x_min'] + obs['x_max']) / 2
        center_z = (obs['z_min'] + obs['z_max']) / 2
        # 플레이어와의 거리 계산
        distance = math.sqrt((center_x - player_pos[0])**2 + (center_z - player_pos[1])**2)
        if distance <= 1200:  # 1200m 이내인 경우만 추가
            valid_enemies.append({
                'x': center_x,
                'z': center_z,
                # 'className': obs['className'],
                'confidence': 1.0,  # /set_obstacles 데이터 신뢰도
                'source': 'obstacles',
                'distance': distance
            })
            logging.debug(f"Valid enemy added: x={center_x:.2f}, z={center_z:.2f}, distance={distance:.2f}m")
        else:
            logging.debug(f"Enemy excluded (too far): x={center_x:.2f}, z={center_z:.2f}, distance={distance:.2f}m")

    if not valid_enemies:
        logging.info("No matching enemies within 1200m")
        return {'message': 'No matching enemy found within 1200m', 'state': False}
    
    # [(75.59507751464844, 93.69336700439453), (66.18401336669922, 115.299461364 7461), (34.20042037963867, 126.11703491210938), (87.6807632446289, 120.0806655883789), (100.6561279296875, 108.7013168334961)]


    min_distance = float('inf')
    nearest_enemy = None
    for enemy in valid_enemies:
        logging.info("😡valid_enemies")
        if enemy['distance'] < min_distance:
            min_distance = enemy['distance']
            nearest_enemy = {
                'x': enemy['x'],
                'z': enemy['z'],
                'y': 10.0,
                'state': True
                # 'distance': enemy['distance'],
                # # 'className': enemy['className'],
                # 'confidence': enemy['confidence'],
                # 'source': enemy['source']
            }
    
    if nearest_enemy:
        logging.debug(f"Nearest enemy: {nearest_enemy}")
    else:
        logging.info("No nearest enemy found after filtering")
        return {'message': 'No valid enemy found within 1200m', 'state': False}
    
    # print('nearest_enemy', nearest_enemy)
    return nearest_enemy