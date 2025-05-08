# modules/is_near_enemy.py
# Ray + Cosine Similarity 방식
import math

def bbox_to_direction_vector(x_pixel, y_pixel, image_width, image_height, fov_h=50, fov_v=28):
    """이미지 상 bbox 중심을 시야 각도 벡터로 변환"""
    print("📐bbox_to_direction_vector_called")
    x_angle = (x_pixel - image_width / 2) / image_width * fov_h
    z_angle = (y_pixel - image_height / 2) / image_height * fov_v
    dir_x = math.tan(math.radians(x_angle))
    dir_z = 1
    norm = math.sqrt(dir_x**2 + dir_z**2)
    return (dir_x / norm, dir_z / norm)

def get_obstacle_center(obstacle):
    print("📐get_obstacle_center_called")
    return ((obstacle['x_min'] + obstacle['x_max']) / 2,
            (obstacle['z_min'] + obstacle['z_max']) / 2)

def cosine_similarity(vec1, vec2):
    print("📐cosine_similarity_called")
    dot = vec1[0]*vec2[0] + vec1[1]*vec2[1]
    norm1 = math.sqrt(vec1[0]**2 + vec1[1]**2)
    norm2 = math.sqrt(vec2[0]**2 + vec2[1]**2)
    return dot / (norm1 * norm2 + 1e-6)

def match_detection_to_obstacle(detection, player_pos, obstacles, image_width, image_height):
    """YOLO bbox 중심과 가장 일치하는 obstacle을 매칭"""
    print("💍match_detection_to_obstacle")
    x_pixel = (detection['bbox'][0] + detection['bbox'][2]) / 2
    y_pixel = (detection['bbox'][1] + detection['bbox'][3]) / 2

    direction = bbox_to_direction_vector(x_pixel, y_pixel, image_width, image_height)
    print(f"\n🎯 Detection center (px): x={x_pixel:.1f}, y={y_pixel:.1f} → direction={direction}")
    
    best_score = -1 # 초기 값 : 장애물과 탐지된 객체가 반대방향에 있음 
    best_obstacle = None
    for obs in obstacles:
        obs_x, obs_z = get_obstacle_center(obs)
        vec_to_obs = (obs_x - player_pos['x'], obs_z - player_pos['z'])
        score = cosine_similarity(direction, vec_to_obs)

        print(f"🔹 Obstacle center: x={obs_x:.2f}, z={obs_z:.2f} → vec_to_obs={vec_to_obs}, score={score:.4f}")

        if score > best_score:
            best_score = score
            best_obstacle = obs

    if best_obstacle:
        center_x, center_z = get_obstacle_center(best_obstacle)
        print(f"✅ Best match → x={center_x:.2f}, z={center_z:.2f}, score={best_score:.4f}")
        return {'x': center_x, 'z': center_z}
    
    print("❌ No obstacle matched.")
    return None
