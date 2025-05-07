import math
import logging

# 로깅 설정
logging.basicConfig(filename='tank.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def match_obstacles_with_detections(obstacles, detections, threshold=120.0):
    """obstacles와 detections 매칭하여 클래스 할당"""
    matched_obstacles = []
    enemy_classes = {'car002', 'car003', 'tank'}
    
    for obs in obstacles:
        obs_center_x = (obs['x_min'] + obs['x_max']) / 2
        obs_center_z = (obs['z_min'] + obs['z_max']) / 2
        matched_class = None
        min_distance = float('inf')
        
        for det in detections:
            if det['className'] not in enemy_classes or det['confidence'] < 0.3:
                continue
            # FIXME : detection이 되게 만들 것
            det_x = det['map_center']['x']
            det_z = det['map_center']['z']
            distance = math.sqrt((obs_center_x - det_x)**2 + (obs_center_z - det_z)**2)
            if distance < min_distance and distance <= threshold:
                min_distance = distance
                matched_class = det['className']
        
        matched_obstacles.append({
            'x_min': obs['x_min'],
            'x_max': obs['x_max'],
            'z_min': obs['z_min'],
            'z_max': obs['z_max'],
            'className': matched_class if matched_class else 'unknown',
            'distance_to_match': min_distance if matched_class else None
        })
        logging.debug(f"Obstacle at x={obs_center_x:.2f}, z={obs_center_z:.2f}: class={matched_class or 'unknown'}, match_distance={min_distance if matched_class else 'N/A'}")
    
    return matched_obstacles

def find_nearest_enemy(detections, player_pose, obstacles, match_threshold=3.0):
    """가장 가까운 적 반환 (120m 이내, detections 우선, obstacles 보조)"""
    logging.debug(f"Starting find_nearest_enemy with {len(detections)} detections, player_pose: {player_pose}, obstacles: {len(obstacles)}")
    
    enemy_classes = {'car002', 'car003', 'tank'}
    valid_enemies = []
    
    # 1. detections에서 적 탐지
    detected_classes = {det['className'] for det in detections if det['className'] in enemy_classes and det['confidence'] >= 0.3}
    logging.debug(f"Detected classes: {detected_classes}")
    
    if not player_pose or 'x' not in player_pose or 'z' not in player_pose:
        logging.warning("Player pose not set or incomplete")
        return {'message': 'Player pose not set'}
    
    for det in detections:
        if det['className'] in enemy_classes and det['confidence'] >= 0.3:
            center_x = det['map_center']['x']
            center_z = det['map_center']['z']
            distance = math.sqrt((center_x - player_pose['x'])**2 + (center_z - player_pose['z'])**2)
            if distance <= 120:
                valid_enemies.append({
                    'x': center_x,
                    'z': center_z,
                    'className': det['className'],
                    'confidence': det['confidence'],
                    'source': 'detections',
                    'distance': distance
                })
                logging.debug(f"Valid enemy (detections): class={det['className']}, x={center_x:.2f}, z={center_z:.2f}, distance={distance:.2f}m")
            else:
                logging.debug(f"Enemy excluded (detections, too far): class={det['className']}, x={center_x:.2f}, z={center_z:.2f}, distance={distance:.2f}m")

    # 2. obstacles 매칭 및 적 탐지
    matched_obstacles = match_obstacles_with_detections(obstacles, detections, threshold=match_threshold)
    for obs in matched_obstacles:
        if obs['className'] in enemy_classes:
            center_x = (obs['x_min'] + obs['x_max']) / 2
            center_z = (obs['z_min'] + obs['z_max']) / 2
            distance = math.sqrt((center_x - player_pose['x'])**2 + (center_z - player_pose['z'])**2)
            if distance <= 120:
                valid_enemies.append({
                    'x': center_x,
                    'z': center_z,
                    'className': obs['className'],
                    'confidence': 1.0,
                    'source': 'obstacles',
                    'distance': distance
                })
                logging.debug(f"Valid enemy (obstacles): class={obs['className']}, x={center_x:.2f}, z={center_z:.2f}, distance={distance:.2f}m")
            else:
                logging.debug(f"Enemy excluded (obstacles, too far): class={obs['className']}, x={center_x:.2f}, z={center_z:.2f}, distance={distance:.2f}m")

    # 3. 가장 가까운 적 선택
    if not valid_enemies:
        logging.info("No valid enemies within 120m")
        return {'message': 'No valid enemy found within 120m'}
    
    nearest_enemy = min(valid_enemies, key=lambda e: e['distance'])
    result = {
        'x': nearest_enemy['x'],
        'z': nearest_enemy['z'],
        'y': 10.0,
        'distance': nearest_enemy['distance'],
        'className': nearest_enemy['className'],
        'confidence': nearest_enemy['confidence'],
        'source': nearest_enemy['source']
    }
    logging.debug(f"Nearest enemy: {result}")
    return result

def get_fire_coordinates(nearest_enemy):
    """가장 가까운 적의 포격 좌표 반환"""
    if not nearest_enemy or 'message' in nearest_enemy:
        logging.warning("No valid enemy for fire coordinates")
        return {'message': 'No valid enemy to fire'}
    coordinates = {'x': nearest_enemy['x'], 'z': nearest_enemy['z']}
    logging.debug(f"Fire coordinates: {coordinates}")
    return coordinates
