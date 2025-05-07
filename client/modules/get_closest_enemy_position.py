import math

#3 FOV 및 카메라 설정
FOV_HORIZONTAL = 50
FOV_VERTICAL = 28
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
MAP_WIDTH = 300
MAP_HEIGHT = 300


def calculate_relative_angle(player_data, obstacle_info):
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
        facing_angle = math.atan2(player_facing['z'], player_facing['x'])  # 라디안
        obstacle_angle = math.atan2(dz, dx)

        # 상대각 (라디안 -> 도)
        relative_angle = math.degrees(obstacle_angle - facing_angle)
        
        # 정규화 (-180 ~ 180)
        if relative_angle > 180:
            relative_angle -= 360
        elif relative_angle < -180:
            relative_angle += 360

        target = obstacle_info[index]
        target['angle'] = relative_angle
        print('🤩', relative_angle)
        
    # print('obstacle_info', obstacle_info)
    return obstacle_info

def get_bbox_angle_x(cx):
    dx = (cx - IMAGE_WIDTH / 2) / (IMAGE_WIDTH / 2)
    angle_x = dx * (FOV_HORIZONTAL / 2)
    return angle_x  # 도 단위


# def match_bbox_to_obstacle(bbox_cx, player_pos, player_facing, obstacles_center):
def match_bbox_to_obstacle(detected_results, player_data, obstacles_center):
    print('🎶🤢detected_results', detected_results)
    print('🎶🤢player_data', player_data)
    print('🎶🤢obstacles_center', obstacles_center)

    

    
    
    # obstacles_angle = calculate_relative_angle(player_data, obstacles_center)
    # print('angle', obstacles_angle)
    return None
    for obs in obstacles_center:
        
        best_match = None
        smallest_diff = float('inf')    
        for index, det in enumerate(detected_results):
            bbox = det['bbox']
            bbox_cx = (bbox[0] + bbox[2])/2

            bbox_angle = get_bbox_angle_x(bbox_cx)
            print('bboxxxxx', bbox_angle)

            diff = abs(bbox_angle - angle)
            if diff < smallest_diff:
                smallest_diff = diff
                best_match = obs
            target = detected_results[index]
            target['center'] = best_match
    return best_match
