import math

# Q/E: 수평 방향 (좌우), 1.0 = 21.35도
# R/F: 수직 방향 (상하), 1.0 = 2.67도

HORIZONTAL_DEGREE_PER_WEIGHT = 21.35  # Q/E
VERTICAL_DEGREE_PER_WEIGHT = 2.67    # R/F

def calculate_angle_diff(target_angle, current_angle):
    """ -180 ~ 180 범위의 최소 회전 각도 계산 """
    diff = (target_angle - current_angle + 180) % 360 - 180
    return diff

def generate_action_command(player_pos, turret_x_angle, turret_y_angle, target_pos):
    action_command = []

    # 벡터 차이로 방향 계산
    dx = target_pos["x"] - player_pos["x"]
    dy = target_pos["y"] - player_pos["y"]
    dz = target_pos["z"] - player_pos["z"]

    # 목표까지의 방향 각도 (수평 기준)
    target_yaw = math.degrees(math.atan2(dx, dz))  # atan2(x, z)
    target_pitch = math.degrees(math.atan2(dy, math.sqrt(dx**2 + dz**2)))

    # 현재 포신 방향 (turret_x는 수평 회전, turret_y는 수직 각도)
    current_yaw = math.degrees(turret_x_angle)
    current_pitch = math.degrees(turret_y_angle)

    # 각도 차이 계산
    yaw_diff = calculate_angle_diff(target_yaw, current_yaw)
    pitch_diff = target_pitch - current_pitch

    # 수평 회전 명령 생성 (Q or E)
    while abs(yaw_diff) > 0.1:
        if yaw_diff > 0:
            direction = "E"
        else:
            direction = "Q"

        weight = min(abs(yaw_diff) / HORIZONTAL_DEGREE_PER_WEIGHT, 1.0)
        action_command.append({"turret": direction, "weight": weight})
        yaw_diff -= math.copysign(weight * HORIZONTAL_DEGREE_PER_WEIGHT, yaw_diff)

    # 수직 포각 조정 명령 (R or F)
    while abs(pitch_diff) > 0.1:
        if pitch_diff > 0:
            direction = "R"
        else:
            direction = "F"

        weight = min(abs(pitch_diff) / VERTICAL_DEGREE_PER_WEIGHT, 1.0)
        action_command.append({"turret": direction, "weight": weight})
        pitch_diff -= math.copysign(weight * VERTICAL_DEGREE_PER_WEIGHT, pitch_diff)

    # 마지막에 발사 명령 추가
    action_command.append({"turret": "FIRE"})

    return action_command
