import math
import logging

# 로깅 설정
logging.basicConfig(filename='tank.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def get_fire_coordinates(nearest_enemy):
    """가장 가까운 적의 포격 좌표 반환"""
    if not nearest_enemy or 'message' in nearest_enemy:
        logging.warning("No valid enemy for fire coordinates")
        return {'message': 'No valid enemy to fire'}
    coordinates = {'x': nearest_enemy['x'], 'z': nearest_enemy['z']}
    logging.debug(f"Fire coordinates: {coordinates}")
    return coordinates

def find_nearest_enemy(detections, player_pos, obstacles):
    """가장 가까운 적 반환 (1200m 이내 적만 valid_enemies로 간주)"""
    logging.debug(f"Starting find_nearest_enemy with {len(detections)} detections, player_pos: {player_pos}, obstacles: {len(obstacles)}")
    
    enemy_classes = {'car002', 'car003', 'tank'}  # 적 클래스
    detected_classes = {det['className'] for det in detections if det['className'] in enemy_classes and det['confidence'] >= 0.3}
    logging.debug(f"Detected classes: {detected_classes}")
    
    if not detected_classes:
        logging.info("No enemy classes detected")
        return {'message': 'No enemy detected'}
    
    if not player_pos:
        logging.warning("Player position not set")
        return {'message': 'Player position not set'}
    
    valid_enemies = []
    for obs in obstacles:
        if obs.get('className') in detected_classes:
            center_x = (obs['x_min'] + obs['x_max']) / 2
            center_z = (obs['z_min'] + obs['z_max']) / 2
            # 플레이어와의 거리 계산
            distance = math.sqrt((center_x - player_pos[0])**2 + (center_z - player_pos[1])**2)
            if distance <= 1200:  # 1200m 이내인 경우만 추가
                valid_enemies.append({
                    'x': center_x,
                    'z': center_z,
                    'className': obs['className'],
                    'confidence': 1.0,  # /set_obstacles 데이터 신뢰도
                    'source': 'obstacles',
                    'distance': distance
                })
                logging.debug(f"Valid enemy added: class={obs['className']}, x={center_x:.2f}, z={center_z:.2f}, distance={distance:.2f}m")
            else:
                logging.debug(f"Enemy excluded (too far): class={obs['className']}, x={center_x:.2f}, z={center_z:.2f}, distance={distance:.2f}m")
    
    # obstacles가 없으면 YOLO bbox 사용
    if not valid_enemies and not obstacles:
        logging.warning("No obstacles data, using YOLO bbox coordinates")
        for det in detections:
            if det['className'] in enemy_classes and det['confidence'] >= 0.3:
                x_center = (det['bbox'][0] + det['bbox'][2]) / 2
                z_center = (det['bbox'][1] + det['bbox'][3]) / 2  # 임시 z 좌표
                distance = math.sqrt((x_center - player_pos[0])**2 + (z_center - player_pos[1])**2)
                if distance <= 1200:  # 1200m 이내인 경우만 추가
                    valid_enemies.append({
                        'x': x_center,
                        'z': z_center,
                        'className': det['className'],
                        'confidence': det['confidence'],
                        'source': 'yolo',
                        'distance': distance
                    })
                    logging.debug(f"Valid enemy added (YOLO): class={det['className']}, x={x_center:.2f}, z={z_center:.2f}, distance={distance:.2f}m")
                else:
                    logging.debug(f"Enemy excluded (YOLO, too far): class={det['className']}, x={x_center:.2f}, z={z_center:.2f}, distance={distance:.2f}m")
    
    logging.debug(f"Valid enemies found: {len(valid_enemies)}")
    if not valid_enemies:
        logging.info("No matching enemies within 1200m")
        return {'message': 'No matching enemy found within 1200m'}
    
    min_distance = float('inf')
    nearest_enemy = None
    for enemy in valid_enemies:
        if enemy['distance'] < min_distance:
            min_distance = enemy['distance']
            nearest_enemy = {
                'x': enemy['x'],
                'z': enemy['z'],
                'y': 10.0,
                'distance': enemy['distance'],
                'className': enemy['className'],
                'confidence': enemy['confidence'],
                'source': enemy['source']
            }
    
    if nearest_enemy:
        logging.debug(f"Nearest enemy: {nearest_enemy}")
    else:
        logging.info("No nearest enemy found after filtering")
        return {'message': 'No valid enemy found within 1200m'}
    
    return nearest_enemy