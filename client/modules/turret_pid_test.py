import math
from modules.pid_controller import PIDController

HORIZONTAL_DEGREE_PER_WEIGHT = 21.35  # 수평: Q/E, w=1.0
VERTICAL_DEGREE_PER_WEIGHT = 2.67    # 수직: R/F, w=1.0

# 사정거리 최대, 최소
MIN_SHOOTING_RANGE = 21
MAX_SHOOTING_RANGE = 128
# 타겟의 거리에 따른 포신의 수직 각도 계산을 위한 상수
PITCH_ESTIMATION_COEFFICIENTS = (-0.0006, 0.2249, -8.6742)

# PID 객체 초기화
yaw_pid = PIDController(kp=1.0, ki=0.0, kd=0.5)
pitch_pid = PIDController(kp=1.5, ki=0.0, kd=0.5)


def calculate_angle_diff(target_angle, current_angle):
    """ -180 ~ 180 범위의 최소 회전 각도 계산 """
    diff = (target_angle - current_angle + 180) % 360 - 180
    return diff

def generate_action_command(type, diff, difault_w):
    action = {}
    # reverse_action_command = []
    if diff > 0:
        direction = "E" if type == 'hor' else "R"
        # reverse_dir = "Q" if type == 'hor' else "F"
    else:
        direction = "Q" if type == 'hor' else "F"
        # reverse_dir = "E" if type == 'hor' else "R"

    weight = min(abs(diff) / difault_w, 1.0)
    action = {"turret": direction, "weight": weight}
    
    return action
    # return (action_command, reverse_action_command)

def get_angles(from_pos, to_pos):
    dx = to_pos['x'] - from_pos['x']
    dy = to_pos['y'] - from_pos['y']
    dz = to_pos['z'] - from_pos['z']

    flat_distance = math.sqrt(dx**2 + dz**2)
    
    yaw = math.degrees(math.atan2(dx, dz))
    a, b, c = PITCH_ESTIMATION_COEFFICIENTS
    pitch = max(min(a * (flat_distance ** 2) + b * flat_distance + c, 10), -5)

    return flat_distance, yaw, pitch

def get_action_command(player_pos, target_pos, hit_pos=None, turret_x_angle=None, turret_y_angle=None, player_y_angle=None):
    print('🐟🐟', player_pos, turret_x_angle, turret_y_angle, player_y_angle, target_pos)
    action_command = []

    flat_distance, target_yaw, target_pitch = get_angles(player_pos, target_pos)

    if hit_pos is None:
        # 첫 번째 발사
        if not (MIN_SHOOTING_RANGE <= flat_distance <= MAX_SHOOTING_RANGE):
            print(f"🤢 [INFO] Target z={flat_distance:.2f} is out of range ({MIN_SHOOTING_RANGE:.2f}~{MAX_SHOOTING_RANGE:.2f}).")
            return {"turret": "Q", "weight": 0.0}
        
        if turret_x_angle is None or turret_y_angle is None or player_y_angle is None:
            raise ValueError("turret_x_angle, turret_y_angle, player_y_angle must be provided if hit_pos is None.")
        current_yaw = turret_x_angle
        current_pitch = turret_y_angle - player_y_angle
        
        yaw_diff = calculate_angle_diff(target_yaw, current_yaw)
        # pitch_diff = target_pitch - turret_y_angle - player_y_angle
        pitch_diff = target_pitch - turret_y_angle -4 # 재조준 테스트 위한 오조준
    else:
        # 재조준
        _, hit_yaw, hit_pitch = get_angles(player_pos, hit_pos)
        current_yaw = hit_yaw
        current_pitch = hit_pitch

        yaw_diff = calculate_angle_diff(target_yaw, current_yaw)
        pitch_diff = calculate_angle_diff(target_pitch, current_pitch)

    yaw_output = yaw_pid.compute(yaw_diff)
    pitch_output = pitch_pid.compute(pitch_diff)
    print("😡", yaw_output, pitch_output)

    # 수평(Q/E) / 수직(R/F)
    if abs(yaw_output) > 0.1:
        action = generate_action_command('hor', yaw_output, HORIZONTAL_DEGREE_PER_WEIGHT)
        # print("hor/😡😡", action)
        action_command.append(action)
    elif abs(pitch_output) > 0.5:
        action = generate_action_command('ver', pitch_output, VERTICAL_DEGREE_PER_WEIGHT)
        # print("ver/😡😡😡", action)
        action_command.append(action)
    else:
        # action = {"turret": "FIRE"}
        action_command.append({"turret": "FIRE"})
        action_command.append({"turret": "Q", "weight": 0.0})

    return action_command

# 명중 확인
def is_hit(target_pos, bullet_pos, tolerance=5.5):
    dx = target_pos["x"] - bullet_pos["x"]
    # dy = target_pos["y"] - bullet_pos["y"]
    dz = target_pos["z"] - bullet_pos["z"]

    distance = math.sqrt(dx ** 2 + dz ** 2)
    is_hit = distance <= tolerance
    return 1 if is_hit else 0
    # return (distance, is_hit)

# 명중 후 turret 원위치
def get_reverse_action_command(turret_x_angle, turret_y_angle, player_y_angle):
    action_command = []
    yaw_diff = calculate_angle_diff(0, turret_x_angle)
    pitch_diff = -turret_y_angle - player_y_angle
    
    # 수평(Q/E) / 수직(R/F)
    hor_action = generate_action_command('hor', yaw_diff, HORIZONTAL_DEGREE_PER_WEIGHT)
    ver_action = generate_action_command('ver', pitch_diff, VERTICAL_DEGREE_PER_WEIGHT)
    action_command = [*hor_action, *ver_action]
    action_command.append({"turret": "Q", "weight": 0.0})
    return action_command