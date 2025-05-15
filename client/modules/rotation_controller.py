import time
import math

def angle_diff(new, old):
    """new - old를 기준으로, 실제 회전한 각도 계산 (0~360 wrap-around 대응)"""
    diff = (new - old + 360) % 360
    return diff if diff < 180 else diff - 360

def start_rotation_10degree(player_data, action_command):
    print("🌀 start_rotation_10degree called")
    total_rotated = 0.0
    prev_angle = player_data.get("turret_x", 0)

    while total_rotated < 360:
        print("🕹️ 회전 명령 발행: Q")
        action_command.append({"turret": "Q", "weight": 0.23})  # 10도 회전
        action_command.append({"turretQE": {"command": "", "weight": 0.0}, "fire": False})# 회전
        time.sleep(3)  # 회전 반영 시간 대기

        curr_angle = player_data.get("turret_x", 0)
        delta = angle_diff(curr_angle, prev_angle)
        total_rotated += abs(delta)
        prev_angle = curr_angle

        print(f"📐 현재 각도: {curr_angle:.2f}°, 누적 회전량: {total_rotated:.2f}°")
        time.sleep(3)

        if total_rotated >= 360:
            print("✅ 360도 회전 완료. 정지 명령 추가.")
            action_command.append({"turret": "STOP", "weight": 0.0})
            break
