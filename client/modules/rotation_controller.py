import time
import math

def angle_diff(new, old):
    """new - old를 기준으로, 실제 회전한 각도 계산 (0~360 wrap-around 대응)"""
    diff = (new - old + 360) % 360
    return diff if diff < 180 else diff - 360

def get_safe_angle(player_data: dict, key: str, fallback: float = 0.0) -> float:
    """
    player_data에서 안전하게 각도 데이터를 가져옴.
    None이거나 형식 이상일 경우 fallback 값 반환.
    """
    try:
        value = player_data.get(key, fallback)
        if value is None:
            print(f"⚠️ Warning: {key} is None, using fallback {fallback}")
            return fallback
        return float(value)
    except Exception as e:
        print(f"❌ Error reading {key}: {e}")
        return fallback

def start_rotation_10degree(get_player_data, action_command):
    print("🌀 start_rotation_10degree called")

    total_rotated = 0.0
    prev_angle = get_player_data().get("turret_x", 0)
    print(f"🐜prev_turret_x: {prev_angle:.2f}")

    while total_rotated < 10:
        print("🕹️ 회전 명령 발행: Q")
        action_command.append({"turretQE": {"command": "Q", "weight": 0.23}, "fire": False})# 10도 회전
        time.sleep(5)  # 회전 반영 시간 대기

        curr_angle = get_player_data().get("turret_x", prev_angle+10)
        print(f"🐜🐜current_turret_x: {curr_angle:.2f}")

        delta = angle_diff(curr_angle, prev_angle)
        print(f"delta : {delta}")

        total_rotated += abs(delta)
        prev_angle = curr_angle

        print(f"📐 현재 각도: {curr_angle:.2f}°, 누적 회전량: {total_rotated:.2f}°")
        
        time.sleep(3)
        
        print("✅ 10도 회전 완료")
        action_command.append({
            "turretQE": {"command": "", "weight": 0.0},
            "fire": False
        })
        break
    return 

def degree_check_and_stop(player_data, action_command,prev_angle, total_rotated):
    print("🌀 degree_check called")
    curr_angle = player_data.get("turret_x", 0)
    delta = angle_diff(curr_angle, prev_angle)
    total_rotated += abs(delta)
    prev_angle = curr_angle

    print("delta : ",delta)
    print(f"📐 현재 각도: {curr_angle:.2f}°, 누적 회전량: {total_rotated:.2f}°")

    if total_rotated >= 360:
        print("✅ 360도 회전 완료. 정지 명령.")
        action_command.append({"turretQE": {"command": "", "weight": 0.0}, "fire": False})
        return True