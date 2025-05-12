import math

HORIZONTAL_DEGREE_PER_WEIGHT = 21.35  # 수평: Q/E, w=1.0
VERTICAL_DEGREE_PER_WEIGHT = 2.67    # 수직: R/F, w=1.0

# 사정거리 최대, 최소
MIN_SHOOTING_RANGE = 21
MAX_SHOOTING_RANGE = 128
# 타겟의 거리에 따른 포신의 수직 각도 계산을 위한 상수
PITCH_ESTIMATION_COEFFICIENTS = (-0.0006, 0.2249, -8.6742)

def calculate_angle_diff(target_angle, current_angle):
    """ -180 ~ 180 범위의 최소 회전 각도 계산 """
    diff = (target_angle - current_angle + 180) % 360 - 180
    return diff

# action_command 생성
def generate_action_command(type, diff, difault_w):
    action_command = []
    while abs(diff) > 0.1:
        if diff > 0:
            direction = "E" if type == 'hor' else "R"
        else:
            direction = "Q" if type == 'hor' else "F"

        weight = min(abs(diff) / difault_w, 1.0)
        action_command.append({"turret": direction, "weight": weight})
        diff -= math.copysign(weight * difault_w, diff)

    return action_command

def get_angles(from_pos, to_pos):
    print("11111111", from_pos)
    dx = to_pos['x'] - from_pos['x']
    dy = to_pos['y'] - from_pos['y']
    dz = to_pos['z'] - from_pos['z']

    flat_distance = math.sqrt(dx**2 + dz**2)
    
    yaw = math.degrees(math.atan2(dx, dz))
    a, b, c = PITCH_ESTIMATION_COEFFICIENTS
    flat_pitch = a * (flat_distance ** 2) + b * flat_distance + c
    adjusted_distance = flat_distance - (dy / math.tan(flat_pitch))
    pitch = a * (adjusted_distance ** 2) + b * adjusted_distance + c

    return flat_distance, yaw, pitch

def get_action_command(player_pos, target_pos, hit_pos=None, turret_x_angle=None, turret_y_angle=None, player_y_angle=None):
    print('🐟🐟', player_pos, turret_x_angle, turret_y_angle, player_y_angle, target_pos)
    action_command = []

    flat_distance, target_yaw, target_pitch = get_angles(player_pos, target_pos)

    if hit_pos is None:
        # 첫 번째 발사
        if not (MIN_SHOOTING_RANGE <= flat_distance <= MAX_SHOOTING_RANGE):
            print(f"🤢 [INFO] Target z={flat_distance:.2f} is out of range ({MIN_SHOOTING_RANGE:.2f}~{MAX_SHOOTING_RANGE:.2f}).")
            return action_command
        
        if turret_x_angle is None or turret_y_angle is None or player_y_angle is None:
            raise ValueError("turret_x_angle, turret_y_angle, player_y_angle must be provided if hit_pos is None.")

        player_y_angle_offset = (turret_y_angle + player_y_angle + 180) % 360 - 180

        yaw_diff = calculate_angle_diff(target_yaw, turret_x_angle)
        pitch_diff = target_pitch - player_y_angle_offset

        # pitch_diff = target_pitch - turret_y_angle - player_y_angle
        # pitch_diff = target_pitch - turret_y_angle # 재조준 테스트 위한 오조준
        print("🤢 pitch_diff", pitch_diff)
        print("🤢 target_pitch", target_pitch)
        print("🤢 turret_y_angle", turret_y_angle)
        print("🤢 player_y_angle_offset", player_y_angle_offset)

    else:
        # 재조준
        _, hit_yaw, hit_pitch = get_angles(player_pos, hit_pos)

        yaw_diff = calculate_angle_diff(target_yaw, hit_yaw)
        pitch_diff = calculate_angle_diff(target_pitch, hit_pitch)

    # 수평(Q/E) / 수직(R/F)
    hor_action = generate_action_command('hor', yaw_diff, HORIZONTAL_DEGREE_PER_WEIGHT)
    ver_action = generate_action_command('ver', pitch_diff, VERTICAL_DEGREE_PER_WEIGHT)

    action_command = [*hor_action, *ver_action]
    action_command.append({"turret": "FIRE"})
    action_command.append({"turret": "Q", "weight": 0.1})

    return action_command

# 명중 확인
def is_hit(target_pos, bullet_pos, tolerance=5.5):
    if not target_pos:
        print("⚠️ is_hit() skipped: target_pos is None")
        return False
    
    print("🤷‍♂️target_pos", target_pos)
    print("🤷bullet_pos", bullet_pos)
    
    # # className으로 명중 판별
    # is_hit = target_pos['className'] == bullet_pos['target'] or bullet_pos['target'] == 'enemy'
    # return is_hit

    # tolerence로 명중 판별
    dx = target_pos["x"] - bullet_pos["x"]
    dz = target_pos["z"] - bullet_pos["z"]

    distance = math.sqrt(dx ** 2 + dz ** 2)
    is_hit = distance <= tolerance
    return is_hit

# 명중 후 turret 원위치
def get_reverse_action_command(turret_x_angle, turret_y_angle, player_x_angle, player_y_angle):
    action_command = []
    yaw_diff = calculate_angle_diff(player_x_angle, turret_x_angle)
    pitch_diff = -turret_y_angle - ((player_y_angle + 180) % 360 - 180)
    
    # 수평(Q/E) / 수직(R/F)
    hor_action = generate_action_command('hor', yaw_diff, HORIZONTAL_DEGREE_PER_WEIGHT)
    ver_action = generate_action_command('ver', pitch_diff, VERTICAL_DEGREE_PER_WEIGHT)
    # action_command = [*hor_action, *ver_action]
    action_command.append({"turret": "Q", "weight": 0.1}) # 도리도리 안하고 360도 로만 돌게 함
    return action_command