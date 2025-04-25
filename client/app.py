from flask import Flask, request, jsonify
import math
import time
import torch
from ultralytics import YOLO
import asyncio
import threading
import requests
import base64
from io import BytesIO
from PIL import Image
import heapq

app = Flask(__name__)
model = YOLO('best.pt')

# ── Node 및 Grid 클래스 ──
class Node:
    def __init__(self, x, z):
        self.x = x
        self.z = z
        self.is_obstacle = False

class Grid:
    def __init__(self, width=300, height=300):
        self.width = width
        self.height = height
        self.grid = [[Node(x, z) for z in range(height)] for x in range(width)]

    def node_from_world_point(self, world_x, world_z):
        gx = max(0, min(int(world_x), self.width-1))
        gz = max(0, min(int(world_z), self.height-1))
        return self.grid[gx][gz]

    def set_obstacle(self, x_min, x_max, z_min, z_max):
        x_min = max(0, min(int(x_min), self.width-1))
        x_max = max(0, min(int(x_max), self.width-1))
        z_min = max(0, min(int(z_min), self.height-1))
        z_max = max(0, min(int(z_max), self.height-1))
        for x in range(x_min, x_max+1):
            for z in range(z_min, z_max+1):
                self.grid[x][z].is_obstacle = True
        print(f"🪨 Grid obstacle set: x_min={x_min}, x_max={x_max}, z_min={z_min}, z_max={z_max}")

    def get_neighbors(self, node):
        dirs = [(-1,0),(1,0),(0,-1),(0,1)]
        result = []
        for dx, dz in dirs:
            nx, nz = node.x+dx, node.z+dz
            if 0 <= nx < self.width and 0 <= nz < self.height and not self.grid[nx][nz].is_obstacle:
                result.append(self.grid[nx][nz])
        return result

# ── 전역 변수 ──
grid = Grid()
obstacles = []

destination = None
current_position = None
last_position = None
last_valid_angle = None
state = "IDLE"
distance_to_destination = float('inf')
rotation_start_time = None
pause_start_time = None
last_body_x = last_body_y = last_body_z = None
last_control = "STOP"
last_weight = 0.0
enemy_detected = False
last_shot_target = None
last_shot_time = 0.0
shot_cooldown = 2.0

# ── 상수 ──
ROTATION_THRESHOLD_DEG = 5
STOP_DISTANCE = 45.0
SLOWDOWN_DISTANCE = 100.0
ROTATION_TIMEOUT = 0.8
PAUSE_DURATION = 0.5
WEIGHT_LEVELS = [1.0, 0.6, 0.3, 0.1, 0.05, 0.01]
DETECTION_RANGE = 100.0

ENEMY_CLASSES    = {'car2','car3','tank'}
FRIENDLY_CLASSES = {'car5'}
OBSTACLE_CLASSES = {'rock1','rock2','wall1','wall2'}

# ── 가중치 계산 ──
def select_weight(val, levels=WEIGHT_LEVELS):
    return min(levels, key=lambda x: abs(x-val))

def calculate_move_weight(dist):
    if dist <= STOP_DISTANCE:
        return 0.0
    if dist > SLOWDOWN_DISTANCE:
        return 1.0
    norm = (dist-STOP_DISTANCE)/(SLOWDOWN_DISTANCE-STOP_DISTANCE)
    target = 0.01 + (1.0-0.01)*(norm**2)
    return select_weight(target)

def calculate_rotation_weight(deg):
    if abs(deg) < ROTATION_THRESHOLD_DEG:
        return 0.0
    target = min(0.3, (abs(deg)/45))
    return select_weight(target)

# ── 장애물 분석 및 사격 ──
async def analyze_obstacle(obs, idx):
    x_c = (obs['x_min']+obs['x_max'])/2
    z_c = (obs['z_min']+obs['z_max'])/2
    img_data = obs.get('image')
    target_map = {0:'car2',1:'car3',2:'car5',3:'human1',4:'rock1',5:'rock2',6:'tank',7:'wall1',8:'wall2'}
    cls = 'unknown'; conf = 0.0
    if img_data:
        try:
            b = base64.b64decode(img_data)
            im = Image.open(BytesIO(b))
            res = model.predict(im, verbose=False)
            dets = res[0].boxes.data.cpu().numpy()
            filtered = []
            for box in dets:
                cid = int(box[5])
                if cid in target_map:
                    filtered.append({'className':target_map[cid], 'bbox':box[:4].tolist(), 'confidence':float(box[4])})
            if filtered:
                best = max(filtered, key=lambda x: x['confidence'])
                cls, conf = best['className'], best['confidence']
        except Exception as e:
            print(f"YOLO fail at ({x_c:.1f},{z_c:.1f}): {e}")
    print(f"Detection at ({x_c:.1f},{z_c:.1f}): {cls} ({conf:.2f})")
    if cls in ENEMY_CLASSES:
        print(f"Enemy: {cls} at ({x_c:.1f},{z_c:.1f})")
    elif cls in FRIENDLY_CLASSES:
        print(f"Friendly: {cls} at ({x_c:.1f},{z_c:.1f})")
    return {'className':cls, 'position':(x_c,z_c)}

async def shoot_at_target(pos):
    global enemy_detected, last_shot_target, last_shot_time
    now = asyncio.get_event_loop().time()
    if now - last_shot_time >= shot_cooldown:
        last_shot_time = now
        last_shot_target = pos
        enemy_detected = True
        data = {'x':pos[0], 'y':0.0, 'z':pos[1], 'hit':'enemy'}
        for i in range(2):
            try:
                r = requests.post('http://localhost:5000/update_bullet', json=data, timeout=5)
                print(f"Shot success: {r.status_code}")
                return
            except:
                continue
        print("All shot attempts failed")

# ── A* 경로 탐색 ──
def a_star(start, goal, grid):
    def h(n, g): return math.hypot(n.x-g.x, n.z-g.z)
    s = grid.node_from_world_point(start[0], start[1])
    g = grid.node_from_world_point(goal[0], goal[1])
    open_set = [(0+h(s,g), 0, s)]
    came, gs, fs = {}, {s:0}, {s:h(s,g)}
    while open_set:
        _, cg, cur = heapq.heappop(open_set)
        if cur == g:
            path=[]
            while cur in came:
                path.append((cur.x,cur.z)); cur=came[cur]
            path.append((s.x,s.z)); path.reverse(); return path
        for nbr in grid.get_neighbors(cur):
            tg = cg+1
            if nbr not in gs or tg < gs[nbr]:
                came[nbr]=cur; gs[nbr]=tg; fs[nbr]=tg+h(nbr,g)
                heapq.heappush(open_set,(fs[nbr], tg, nbr))
    return []

# ── 이동 및 장애물 처리 태스크 ──
async def move_towards_destination():
    global current_position, state
    while destination and state not in ["STOPPED","IDLE"]:
        pos = current_position; dest = destination
        dist = math.hypot(dest[0]-pos[0], dest[1]-pos[1])
        if dist < STOP_DISTANCE:
            state = "STOPPED"; break
        path = a_star(pos, dest, grid)
        if not path:
            state = "STOPPED"; break
        nxt = path[1] if len(path)>1 else dest
        current_position = nxt
        # 장애물 탐지 및 사격
        for obs in obstacles:
            oc = ((obs['x_min']+obs['x_max'])/2, (obs['z_min']+obs['z_max'])/2)
            od = math.hypot(oc[0]-pos[0], oc[1]-pos[1])
            if od < DETECTION_RANGE:
                det = await analyze_obstacle(obs,0)
                if det['className'] in ENEMY_CLASSES:
                    await shoot_at_target(oc)
        await asyncio.sleep(0.1)
    print("🏁 move_towards_destination stopped")

def run_async_task():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(move_towards_destination())
    loop.close()

# ── API 엔드포인트 ──
@app.route('/detect', methods=['POST'])
# (위 analyze_obstacle 기능 대신 간단 YOLO detect)
def detect(): pass  # 이미 정의됨 위로 이동

@app.route('/info', methods=['POST'])
def info():
    global state, destination, current_position, last_position, distance_to_destination
    global rotation_start_time, pause_start_time, last_valid_angle
    global last_body_x, last_body_y, last_body_z, last_control, last_weight

    data = request.get_json(force=True)
    if not data:
        return jsonify({'error': 'No JSON received'}), 400
    if not destination:
        state = "IDLE"; last_control, last_weight = "STOP",0.0
        return jsonify(status="success", control="STOP", weight=0.0)

    # 파싱
    p = data.get('playerPos', {})
    bodyX, bodyY, bodyZ = data.get('playerBodyX',0.0),data.get('playerBodyY',0.0),data.get('playerBodyZ',0.0)
    distance_to_destination = data.get('distance', float('inf'))
    current_position = (p.get('x',0.0), p.get('z',0.0))

    # 방향 보정
    if last_position and current_position!=last_position:
        dx, dz = current_position[0]-last_position[0], current_position[1]-last_position[1]
        if math.hypot(dx,dz)>1e-4: current_angle=math.atan2(dz,dx)
        else: current_angle=math.radians(bodyX)
    else:
        dx,dz = destination; px,pz = current_position
        current_angle = math.atan2(dz-pz, dx-px)
    last_valid_angle = current_angle

    # 바디 변화 로그
    if last_body_x is not None:
        dbx, dby, dbz = bodyX-last_body_x, bodyY-last_body_y, bodyZ-last_body_z
        if abs(dbx)<1e-3 and state=="ROTATING": print("⚠️ bodyX change too small during ROTATING")
        print(f"🔄 Δbody: X={dbx:.3f}, Y={dby:.3f}, Z={dbz:.3f}")
    last_body_x, last_body_y, last_body_z = bodyX, bodyY, bodyZ

    # FSM
    control, weight = "STOP", 0.0
    if state=="IDLE":
        state="ROTATING"; rotation_start_time=time.time()
    elif state=="ROTATING":
        # 벡터 연산
        fx, fz = math.cos(current_angle), math.sin(current_angle)
        tx, tz = destination[0]-current_position[0], destination[1]-current_position[1]
        d = math.hypot(tx,tz)
        if d>1e-6: tx/=d; tz/=d
        dot = max(-1, min(1, fx*tx + fz*tz))
        deg = math.degrees(math.acos(dot))
        cross = fx*tz - fz*tx
        print(f"🧭 ROTATING: angle_diff={deg:.2f}°, cross={cross:.3f}")
        if rotation_start_time and (time.time()-rotation_start_time)>ROTATION_TIMEOUT:
            state="PAUSE"; pause_start_time=time.time()
        elif deg<ROTATION_THRESHOLD_DEG:
            state="PAUSE"; pause_start_time=time.time()
        else:
            control = "A" if cross>0 else "D"
            weight = calculate_rotation_weight(deg)
    elif state=="PAUSE":
        if time.time()-pause_start_time>=PAUSE_DURATION:
            state="MOVING"; control="W"; weight=calculate_move_weight(distance_to_destination)
            threading.Thread(target=run_async_task, daemon=True).start()
    elif state=="MOVING":
        fx, fz = math.cos(current_angle), math.sin(current_angle)
        tx, tz = destination[0]-current_position[0], destination[1]-current_position[1]
        d = math.hypot(tx,tz)
        if d>1e-6: tx/=d; tz/=d
        dot = max(-1, min(1, fx*tx + fz*tz))
        deg = math.degrees(math.acos(dot))
        cross = fx*tz - fz*tx
        if distance_to_destination<=STOP_DISTANCE:
            state="STOPPED"
        elif abs(deg)>ROTATION_THRESHOLD_DEG*6:
            state="ROTATING"; rotation_start_time=time.time()
            control = "A" if cross>0 else "D"; weight = calculate_rotation_weight(deg)
        else:
            control="W"; weight=calculate_move_weight(distance_to_destination)
    else:
        control, weight = "STOP",0.0

    last_control, last_weight = control, weight
    last_position = current_position
    return jsonify(status="success", control=control, weight=weight)

@app.route('/set_destination', methods=['POST'])
def set_destination():
    global destination, state, rotation_start_time, last_position, last_valid_angle
    data = request.get_json()
    if not data or "destination" not in data:
        return jsonify({'status':'ERROR','message':'Missing destination'}),400
    try:
        x,y,z = map(float, data['destination'].split(','))
        destination = (x,z)
        last_position, last_valid_angle = None, None
        state = "ROTATING"; rotation_start_time = time.time()
        print(f"🎯 New destination: {x},{y},{z} (reset last_position)")
        return jsonify(status="OK", destination={'x':x,'y':y,'z':z})
    except Exception as e:
        return jsonify({'status':'ERROR','message':str(e)}),400

@app.route('/update_position', methods=['POST'])
def update_position():
    global current_position, last_position, state, destination
    data = request.get_json()
    if not data or "position" not in data:
        return jsonify({'status':'ERROR','message':'Missing position data'}),400
    try:
        x,y,z = map(float,data['position'].split(','))
        current_position = (x,z)
        if last_position:
            dx, dz = x-last_position[0], z-last_position[1]
            print(f"📍 Movement change: dx={dx:.6f}, dz={dz:.6f}")
        if destination:
            dx, dz = destination; z_diff = abs(z-dz)
            ang = math.degrees(math.atan2(dz-z, dx-x))
            print(f"📍 Position updated: {current_position}, target angle: {ang:.2f}°, z_diff: {z_diff:.2f}m")
        else:
            print(f"📍 Position updated: {current_position}")
        return jsonify(status="OK", current_position=current_position)
    except Exception as e:
        return jsonify({'status':'ERROR','message':str(e)}),400

@app.route('/get_move', methods=['GET'])
def get_move():
    return jsonify(move=last_control, weight=last_weight)

@app.route('/get_action', methods=['GET'])
def get_action():
    global enemy_detected
    if enemy_detected:
        enemy_detected = False
        return jsonify(turret="FIRE", weight=1.0)
    return jsonify(turret="", weight=0.0)

@app.route('/update_bullet', methods=['POST'])
def update_bullet():
    data = request.get_json()
    if not data:
        return jsonify({'status':'ERROR','message':'Invalid data'}),400
    print(f"💥 Bullet Impact at X={data.get('x')},Y={data.get('y')},Z={data.get('z')},hit={data.get('hit')}")
    return jsonify(status="OK", message="Bullet impact data received")

@app.route('/update_obstacle', methods=['POST'])
def update_obstacle():
    global obstacles, grid
    data = request.get_json()
    if not data:
        return jsonify({'status':'error','message':'No data received'}),400
    obstacles = data.get('obstacles', [])
    grid = Grid()
    for obs in obstacles:
        det = asyncio.run(analyze_obstacle(obs, 0))
        if det['className'] in OBSTACLE_CLASSES:
            grid.set_obstacle(obs['x_min'], obs['x_max'], obs['z_min'], obs['z_max'])
    print("🪨 Obstacle Data:", obstacles)
    return jsonify(status='success', message='Obstacle data received')

@app.route('/init', methods=['GET'])
def init():
    cfg = {"startMode":"start","blStartX":60,"blStartY":10,"blStartZ":27.23,
           "rdStartX":59,"rdStartY":10,"rdStartZ":280}
    print("🛠️ Initialization config:", cfg)
    return jsonify(cfg)

@app.route('/start', methods=['GET'])
def start():
    print("🚀 /start command received")
    return jsonify(control="")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
