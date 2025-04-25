# movement.py
import asyncio
import threading
import math
import requests
from path_finding import a_star
from detection import analyze_obstacle
from config import STOP_DISTANCE, DETECTION_RANGE, SHOOTING_RANGE, ENEMY_CLASSES

async def shoot_at_target(target_pos, player_state, state_lock):
    with state_lock:
        current_time = asyncio.get_event_loop().time()
        time_since_last_shot = current_time - player_state["last_shot_time"]
        if time_since_last_shot >= player_state["shot_cooldown"]:
            player_state["last_shot_time"] = current_time
            player_state["last_shot_target"] = target_pos
            player_state["enemy_detected"] = True
            bullet_data = {
                "x": target_pos[0],
                "y": 0.0,
                "z": target_pos[1],
                "hit": "enemy"
            }
            print(f"Attempting to shoot at {target_pos}, bullet: {bullet_data}, time_since_last_shot={time_since_last_shot:.2f}s")
            for attempt in range(2):
                try:
                    response = requests.post('http://localhost:5000/update_bullet', json=bullet_data, timeout=5)
                    print(f"Shot fired successfully at {target_pos}, status={response.status_code}")
                    return bullet_data
                except requests.RequestException as e:
                    print(f"Shot failed at {target_pos}: HTTP error on attempt {attempt+1}, error={e}")
            print(f"Shot failed at {target_pos}: all HTTP attempts failed")
            return None
        else:
            print(f"Shot failed at {target_pos}: cooldown active, {player_state['shot_cooldown'] - time_since_last_shot:.2f}s remaining")
            return None

async def move_towards_destination(obstacles, grid, player_state, state_lock):
    print("🚀 Starting move_towards_destination")
    while player_state["destination"] and player_state["state"] not in ["STOPPED", "IDLE"]:
        with state_lock:
            current_pos = player_state["position"]
            dest = player_state["destination"]
        if not (isinstance(current_pos, tuple) and isinstance(dest, tuple) and
                len(current_pos) == 2 and len(dest) == 2):
            print(f"🚫 Invalid position or destination: current_pos={current_pos}, dest={dest}")
            with state_lock:
                player_state["state"] = "STOPPED"
            break

        distance_to_dest = math.sqrt((dest[0] - current_pos[0])**2 + (dest[1] - current_pos[1])**2)
        print(f"🚗 Current position: {current_pos}, destination: {dest}, distance: {distance_to_dest:.2f}, state: {player_state['state']}")

        if distance_to_dest < STOP_DISTANCE:
            with state_lock:
                player_state["position"] = dest
                player_state["state"] = "STOPPED"
            print(f"🎯 Reached destination: {dest}")
            break

        try:
            path = a_star(current_pos, dest, grid)
        except Exception as e:
            print(f"🚫 A* failed: current_pos={current_pos}, dest={dest}, error={e}")
            with state_lock:
                player_state["state"] = "STOPPED"
            break

        if not path:
            with state_lock:
                player_state["state"] = "STOPPED"
            print("🚫 Stopping: no valid A* path to destination")
            break

        next_pos = path[1] if len(path) > 1 else dest
        distance = math.sqrt((next_pos[0] - current_pos[0])**2 + (next_pos[1] - current_pos[1])**2)
        print(f"🚗 Moving to waypoint: {next_pos}, current position: {current_pos}, distance: {distance:.2f}")
        
        if distance < 1.0:
            with state_lock:
                player_state["position"] = next_pos
            print(f"✅ Reached waypoint: {next_pos}")
            continue

        for idx, obstacle in enumerate(obstacles):
            obs_center = ((obstacle["x_min"] + obstacle["x_max"]) / 2, (obstacle["z_min"] + obstacle["z_max"]) / 2)
            obs_distance = math.sqrt((obs_center[0] - current_pos[0])**2 + (obs_center[1] - current_pos[1])**2)
            if obs_distance < DETECTION_RANGE:
                detection = await analyze_obstacle(obstacle, idx)
                class_name = detection["className"]
                if class_name in ENEMY_CLASSES and obs_distance <= SHOOTING_RANGE:
                    await shoot_at_target(obs_center, player_state, state_lock)
                    await asyncio.sleep(0.5)
                elif class_name in ENEMY_CLASSES:
                    print(f"Enemy detected but out of range: {class_name} at ({obs_center[0]:.2f}, {obs_center[1]:.2f}), distance={obs_distance:.2f}m")

        await asyncio.sleep(0.1)
    print("🏁 move_towards_destination stopped")

def run_async_task(obstacles, grid, player_state, state_lock):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(move_towards_destination(obstacles, grid, player_state, state_lock))
    except Exception as e:
        print(f"🚫 Async task failed: error={e}")
    finally:
        loop.close()