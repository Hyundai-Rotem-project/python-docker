import json
import os

def load_obstacles_from_map(map_path):
    if not os.path.exists(map_path):
        print(f"❌ Map file not found: {map_path}")
        return []

    with open(map_path, 'r') as f:
        map_data = json.load(f)

    for index, item in enumerate(map_data['obstacles']):
        item['id'] = f"{item['prefabName']}_{index}"

    print('map_data["obstacles"]', map_data['obstacles'])
    return map_data['obstacles']