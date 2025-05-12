import math
import logging
from sklearn.neighbors import KDTree
import numpy as np

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
        if 'center' not in target:
            target['center'] = (xc, zc)
        target['center'] = (xc, zc)
        # print('🤩', index, relative_angle, (xc, zc))
        
    return obstacle_info

def calculate_bbox_angle(bbox,turret_y):
    """
    YOLO bbox를 기반으로 수평 각도 계산 + turret_y 보정
    """
    bbox_cx = (bbox[0] + bbox[2])/2
    dx = (bbox_cx - IMAGE_WIDTH / 2) / (IMAGE_WIDTH / 2)
    raw_angle = dx * (FOV_HORIZONTAL / 2)
    # turret_y를 기준으로 보정된 실제 시야 방향
    corrected_angle = (turret_y + raw_angle + 360) % 360
    return corrected_angle

def match_bbox_to_obstacle(detected_results, player_data, obstacle_data, top_k=3):
    print("📌 match_bbox_to_obstacle_called")

    if not obstacle_data:
        print("⚠️ No obstacles provided.")
        return detected_results

    # 1. 각도 계산
    obstacle_info = calculate_relative_angle(player_data, obstacle_data)
    turret_y = player_data.get('turret_y', 0)

    # 2. obstacle center 좌표 준비 for KDTree
    centers = np.array([
        [(obs['x_min'] + obs['x_max']) / 2, (obs['z_min'] + obs['z_max']) / 2]
        for obs in obstacle_info
    ])
    kd_tree = KDTree(centers)

    for det in detected_results:
        bbox = det['bbox']
        bbox_angle = calculate_bbox_angle(bbox, turret_y)

        # 3. 각도 기반 후보군 필터링
        candidates = []
        for obs in obstacle_info:
            obs_angle = obs.get('angle')
            if obs_angle is None:
                continue
            angle_diff = abs((obs_angle - bbox_angle + 180) % 360 - 180)
            if angle_diff < 20:  # 각도 차이 기준 필터링
                candidates.append(obs)
        print(f"📐📐bbox_angle: {bbox_angle}, obs_angle: {obs_angle}, angle_diff: {angle_diff}")

        if not candidates:
            print(f"⚠️ No angle-based matches for Detection ID {det.get('id')}")
            continue

        # 4. 각도 후보 중 center로 KDTree 거리 기반 좁히기
        bbox_cx = (bbox[0] + bbox[2]) / 2
        bbox_cz = (bbox[1] + bbox[3]) / 2
        dists, indices = kd_tree.query([[bbox_cx, bbox_cz]], k=top_k)

        # KDTree 후보 중 각도 필터된 것과 일치하는 것 우선 선택
        best_match = None
        for idx in indices[0]:
            candidate = obstacle_info[idx]
            if candidate in candidates:
                best_match = candidate
                break
            
        # 🔁 fallback: KDTree 안에서 실패하면 angle 후보 중 거리 가장 가까운 애라도 쓰기
        if not best_match and candidate:
            # fallback: candidates 중 가장 가까운 것 선택
            min_dist = float('inf')
            for cand in candidates:
                cx = (cand['x_min'] + cand['x_max']) / 2
                cz = (cand['z_min'] + cand['z_max']) / 2
                dist = math.sqrt((bbox_cx - cx) ** 2 + (bbox_cz - cz) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    best_match = cand
            print(f"🆗 Fallback match used for Detection ID {det.get('id')} with distance {min_dist:.2f}")
            print(f"⚠️ No KDTree-refined match for Detection ID {det.get('id')}")
            continue

        # 매칭 성공
        cx = (best_match['x_min'] + best_match['x_max']) / 2
        cz = (best_match['z_min'] + best_match['z_max']) / 2
        det['center'] = (cx, cz)
        det['matched_class'] = best_match.get('className', 'unknown')

        print(f"✅ Detection {det.get('id')} matched with obstacle at "
              f"{det['center']} (angle_diff ≈ {angle_diff:.2f}°)")

    return detected_results


# 🎶🤢detected_results 예시
# {'id': 0, 'className': 'car003', 
#  'bbox': [729.469482421875, 545.0945434570312, 865.4412231445312, 665.6178588867188], 
#  'confidence': 0.8793855905532837, 'color': '#0000FF', 
# 'filled': False, 'updateBoxWhileMoving': False, 
# 'center': (56.05139923095703, 99.60943603515625)}

from sklearn.neighbors import KDTree
import numpy as np

def narrow_obstacles_by_kdtree(detected_results, obstacle_list, top_k=3):
    # obstacle center 좌표 추출
    centers = np.array([
        [(obs['x_min'] + obs['x_max']) / 2, (obs['z_min'] + obs['z_max']) / 2]
        for obs in obstacle_list
    ])
    tree = KDTree(centers)

    for det in detected_results:
        # bbox 중심 계산
        bbox = det['bbox']
        cx = (bbox[0] + bbox[2]) / 2
        cz = (bbox[1] + bbox[3]) / 2  # Y축이 아닌 Z축일 경우 적절히 수정
        dist, idx = tree.query([[cx, cz]], k=top_k)

        # 가장 가까운 후보들 저장
        det['kd_matches'] = [obstacle_list[i] for i in idx[0]]
        det['kd_distances'] = dist[0]

    return detected_results


def find_nearest_enemy(detections, player_data, obstacles):
    """가장 가까운 적 반환 (1200m 이내 적만 valid_enemies로 간주)"""
    print(f"🧪 [find_nearest_enemy] Called with {len(detections)} detections.")
    print("🔎🔎detections", detections)
    for i, det in enumerate(detections):
        print(f"🔎 Detection {i}: class={det['className']}, conf={det['confidence']:.2f}, matched_class={det.get('matched_class')}, center={det.get('center')}")
    
    detected_results = match_bbox_to_obstacle(detections, player_data, obstacles)
    # print('🎶🤢 player_data', player_data)
    # nearest_enemy = find_nearest_enemy(filtered_results, player_data['pos'], obstacles)
    player_pos = player_data['pos']
    logging.debug(f"Starting find_nearest_enemy with {len(detections)} detections, player_pos: {player_pos}, obstacles: {len(obstacles)}")
    
    enemy_classes = {'car002', 'tank'}  # 적 클래스
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
    for detected in detected_results:
        if 'center' not in detected:
            print(f"⚠️ Center missing for detection {det['id']}")
            continue  # center 없는 객체는 건너뜀
        matched_class = detected.get('matched_class', '')
        if matched_class in enemy_classes and detected['confidence'] >= 0.3:
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

def find_all_valid_enemies(detections, player_data, obstacles, max_distance=1200):
    detected_results = match_bbox_to_obstacle(detections, player_data, obstacles)
    player_pos = player_data['pos']
    enemy_classes = {'car002', 'tank'}
    
    valid_enemies = []
    for det in detected_results:
        if 'center' not in det:
            continue
        matched_cls = det.get('matched_class', '').lower()
        if matched_cls not in enemy_classes:
            continue
        if det.get('confidence',0) < 0.3:
            continue

        cx, cz = det['center']
        distance = math.sqrt((cx - player_pos['x'])**2 + (cz - player_pos['z'])**2)
        if distance <= max_distance:
            valid_enemies.append({
                'x': cx,
                'z': cz,
                'y': 8,
                'distance': distance,
                'className': det['className'],
                'state': True,
            })
    
    # 거리순 정렬
    valid_enemies.sort(key=lambda e: e['distance'])
    return valid_enemies
