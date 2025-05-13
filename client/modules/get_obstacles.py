import json
import os

def load_map_to_obstacles(map_path):
    if not os.path.exists(map_path):
        print(f"❌ Map file not found: {map_path}")
        return []

    with open(map_path, 'r') as f:
        map_data = json.load(f)
        
    return map_data['obstacles']