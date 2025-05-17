import json
import os

obstacle_list = [ ]

def load_obstacles(mpath):
    '''Load obstacles from a JSON file and return a list of obstacle coordinates.
    ex    # obstacle_list=[(0, 0), (1, 1)]'''
    with open(mpath, 'r') as f:
        data = json.load(f)
        print(json.dumps(data, indent=2))

    for obs in data["obstacles"]:
        obstacle_x = obs['position']['x'] 
        obstacle_z = obs['position']['z']
        print(obstacle_x, obstacle_z)
        obstacle_list.append((obstacle_x, obstacle_z))
    return obstacle_list
