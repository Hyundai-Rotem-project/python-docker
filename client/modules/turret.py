import math

# Q/E: 수평 방향 (좌우), 1.0 = 21.35도
# R/F: 수직 방향 (상하), 1.0 = 2.67도

HORIZONTAL_DEGREE_PER_WEIGHT = 21.35  # Q/E
VERTICAL_DEGREE_PER_WEIGHT = 2.67    # R/F

MIN_TARGET_Z = 21
MAX_TARGET_Z = 128

def calculate_angle_diff(target_angle, current_angle):
    """ -180 ~ 180 범위의 최소 회전 각도 계산 """
    diff = (target_angle - current_angle + 180) % 360 - 180
    return diff

def add_action_command(type, diff, difault_w):
    action_command = []
    reverse_action_command = []
    while abs(diff) > 0.1:
        if diff > 0:
            direction = "E" if type == 'hor' else "R"
            reverse_dir = "Q" if type == 'hor' else "F"
        else:
            direction = "Q" if type == 'hor' else "F"
            reverse_dir = "E" if type == 'hor' else "R"

        weight = min(abs(diff) / difault_w, 1.0)
        action_command.append({"turret": direction, "weight": weight})
        reverse_action_command.append({"turret": reverse_dir, "weight": weight})
        diff -= math.copysign(weight * difault_w, diff)

    return (action_command, reverse_action_command)

def generate_action_command(player_pos, turret_x_angle, turret_y_angle, target_pos):
    print('🐟🐟', player_pos, turret_x_angle, turret_y_angle, target_pos)
    action_command = []

    # 벡터 차이로 방향 계산
    dx = target_pos["x"] - player_pos["x"]
    dy = target_pos["y"] - player_pos["y"]
    dz = target_pos["z"] - player_pos["z"]

    
    # 🎯 사정거리 조건 확인
    range_limit = math.sqrt(dx**2 + dz**2)
    if not (MIN_TARGET_Z <= range_limit <= MAX_TARGET_Z):
        print(f"🤢 [INFO] Target z={range_limit:.2f} is out of range ({MIN_TARGET_Z:.2f}~{MAX_TARGET_Z:.2f}).")
        return action_command

    # 목표까지의 방향 각도 (수평 기준)
    target_yaw = math.degrees(math.atan2(dx, dz))  # atan2(x, z)
    target_pitch = math.degrees(math.atan2(dy, math.sqrt(dx**2 + dz**2)))

    # 각도 차이 계산
    yaw_diff = calculate_angle_diff(target_yaw, turret_x_angle)
    pitch_diff = target_pitch - turret_y_angle

    # 수평(Q/E) / 수직(R/F)
    hor_action, reverse_hor_action = add_action_command('hor', yaw_diff, HORIZONTAL_DEGREE_PER_WEIGHT)
    ver_action, reverse_ver_action = add_action_command('ver', pitch_diff, VERTICAL_DEGREE_PER_WEIGHT)
    action_command = [*hor_action, *ver_action]
    reverse_action_command = [*reverse_hor_action, *reverse_ver_action]

    # 마지막에 발사 명령 추가
    action_command.append({"turret": "FIRE"})
    action_command.extend(reverse_action_command)
    action_command.append({"turret": "Q", "weight": 0.0})

    return action_command
