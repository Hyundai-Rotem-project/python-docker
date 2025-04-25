# grid.py
import math
from config import OBSTACLE_BUFFER

class Node:
    def __init__(self, x, z):
        self.x = x
        self.z = z
        self.is_obstacle = False

    def __lt__(self, other):
        """Node 비교를 위한 메서드: (x, z) 좌표를 lexicographically 비교"""
        if not isinstance(other, Node):
            return NotImplemented
        return (self.x, self.z) < (other.x, other.z)

    def __eq__(self, other):
        """Node 동일성 비교"""
        if not isinstance(other, Node):
            return False
        return (self.x, self.z) == (other.x, other.z)

    def __repr__(self):
        return f"Node({self.x}, {self.z})"

class Grid:
    def __init__(self, width=300, height=300):
        self.width = width
        self.height = height
        self.grid = [[Node(x, z) for z in range(height)] for x in range(width)]

    def node_from_world_point(self, world_x, world_z):
        try:
            grid_x = max(0, min(int(world_x), self.width - 1))
            grid_z = max(0, min(int(world_z), self.height - 1))
            print(f"🗺️ Node created: world=({world_x:.2f}, {world_z:.2f}) -> grid=({grid_x}, {grid_z})")
            return self.grid[grid_x][grid_z]
        except (TypeError, ValueError) as e:
            print(f"🗺️ Node creation failed: world=({world_x}, {world_z}), error={e}")
            raise ValueError(f"Invalid world coordinates: ({world_x}, {world_z})")

    def set_obstacle(self, x_min, x_max, z_min, z_max):
        # 장애물 주변 5m(5칸) 버퍼 추가
        buffer = int(OBSTACLE_BUFFER)
        x_min = max(0, min(int(x_min) - buffer, self.width - 1))
        x_max = max(0, min(int(x_max) + buffer, self.width - 1))
        z_min = max(0, min(int(z_min) - buffer, self.height - 1))
        z_max = max(0, min(int(z_max) + buffer, self.height - 1))
        for x in range(x_min, x_max + 1):
            for z in range(z_min, z_max + 1):
                self.grid[x][z].is_obstacle = True
        print(f"🪨 Grid obstacle set with buffer: x_min={x_min}, x_max={x_max}, z_min={z_min}, z_max={z_max}")

    def get_neighbors(self, node):
        x, z = node.x, node.z
        neighbors = []
        for dx, dz in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, nz = x + dx, z + dz
            if 0 <= nx < self.width and 0 <= nz < self.height:
                if not self.grid[nx][nz].is_obstacle:
                    neighbors.append(self.grid[nx][nz])
        return neighbors