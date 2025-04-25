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

def add_action_command(type, diff, difault_w):
    action_command = []
    # reverse_action_command = []
    while abs(diff) > 0.1:
        if diff > 0:
            direction = "E" if type == 'hor' else "R"
            # reverse_dir = "Q" if type == 'hor' else "F"
        else:
            direction = "Q" if type == 'hor' else "F"
            # reverse_dir = "E" if type == 'hor' else "R"

        weight = min(abs(diff) / difault_w, 1.0)
        action_command.append({"turret": direction, "weight": weight})
        # reverse_action_command.append({"turret": reverse_dir, "weight": weight})
        diff -= math.copysign(weight * difault_w, diff)

    return action_command
    # return (action_command, reverse_action_command)

def is_hit(target_pos, bullet_pos, tolerance=5.5):
    dx = target_pos["x"] - bullet_pos["x"]
    # dy = target_pos["y"] - bullet_pos["y"]
    dz = target_pos["z"] - bullet_pos["z"]

    distance = math.sqrt(dx ** 2 + dz ** 2)
    is_hit = distance <= tolerance
    return is_hit
    # return (distance, is_hit)

def get_angles(from_pos, to_pos):
    """
    from_pos에서 to_pos를 향한 yaw, pitch 각도 계산
    """
    dx = to_pos['x'] - from_pos['x']
    dy = to_pos['y'] - from_pos['y']
    dz = to_pos['z'] - from_pos['z']

    flat_distance = math.sqrt(dx**2 + dz**2)
    yaw = math.degrees(math.atan2(dx, dz))         # 좌우
    pitch = math.degrees(math.atan2(dy, flat_distance))  # 상하

    return yaw, pitch

def adjust_gun_angle(tank_pos, hit_pos, target_pos):
    """
    포신의 yaw, pitch를 타겟에 정확히 맞도록 조절하는 각도 차이 계산

    Returns:
    - yaw_adjustment: 수평 방향 조정 각도 (deg)
    - pitch_adjustment: 상하 방향 조정 각도 (deg)
    """
    print("???🐟🐟???", tank_pos, hit_pos, target_pos)
    # 전차 → 포탄이 실제 터진 위치
    hit_yaw, hit_pitch = get_angles(tank_pos, hit_pos)

    # 전차 → 타겟 위치
    target_yaw, target_pitch = get_angles(tank_pos, target_pos)

    # 실제 포신 방향 → 타겟을 맞추기 위해 조정해야 할 각도
    yaw_adjustment = target_yaw - hit_yaw
    pitch_adjustment = target_pitch - hit_pitch


    # 수평(Q/E) / 수직(R/F)
    hor_action = add_action_command('hor', yaw_adjustment, HORIZONTAL_DEGREE_PER_WEIGHT)
    ver_action = add_action_command('ver', pitch_adjustment, VERTICAL_DEGREE_PER_WEIGHT)
    # hor_action, reverse_hor_action = add_action_command('hor', yaw_diff, HORIZONTAL_DEGREE_PER_WEIGHT)
    # ver_action, reverse_ver_action = add_action_command('ver', pitch_diff, VERTICAL_DEGREE_PER_WEIGHT)
    action_command = [*hor_action, *ver_action]

    action_command.append({"turret": "FIRE"})
    action_command.append({"turret": "Q", "weight": 0.0})

    return action_command
