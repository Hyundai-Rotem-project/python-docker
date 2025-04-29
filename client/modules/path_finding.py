# path_finding.py
import heapq
import math
from modules.grid import Node, Grid

def a_star(start, goal, grid):
    def heuristic(node, goal_node):
        try:
            if not (isinstance(node.x, (int, float)) and isinstance(node.z, (int, float)) and
                    isinstance(goal_node.x, (int, float)) and isinstance(goal_node.z, (int, float))):
                raise ValueError("Invalid node coordinates")
            distance = math.sqrt((node.x - goal_node.x) ** 2 + (node.z - goal_node.z) ** 2)
            if math.isnan(distance) or math.isinf(distance):
                raise ValueError(f"Invalid heuristic distance: {distance}")
            print(f"🛤️ Heuristic: node=({node.x}, {node.z}), goal_node=({goal_node.x}, {goal_node.z}), distance={distance:.2f}")
            return distance
        except Exception as e:
            print(f"🛤️ Heuristic error: node=({node.x}, {node.z}), goal_node=({goal_node.x}, {goal_node.z}), error={e}")
            raise

    try:
        start_node = grid.node_from_world_point(start[0], start[1])
        goal_node = grid.node_from_world_point(goal[0], goal[1])
        print(f"🛤️ A* calculating path from ({start_node.x}, {start_node.z}) to ({goal_node.x}, {goal_node.z})")
    except Exception as e:
        print(f"🛤️ A* failed to initialize: start={start}, goal={goal}, error={e}")
        return []

    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start_node, goal_node), 0, start_node))
    came_from = {}
    g_score = {start_node: 0}
    f_score = {start_node: heuristic(start_node, goal_node)}

    while open_set:
        try:
            f_score_val, current_g, current = heapq.heappop(open_set)
            print(f"🛤️ Processing node: ({current.x}, {current.z}), f_score={f_score_val:.2f}, g_score={current_g}")
        except IndexError:
            print(f"🛤️ A* failed: open_set is empty")
            return []
        except Exception as e:
            print(f"🛤️ A* failed to pop from open_set: error={e}")
            return []

        if current == goal_node:
            path = []
            while current in came_from:
                path.append((current.x, current.z))
                current = came_from[current]
            path.append((start_node.x, start_node.z))
            path.reverse()
            print(f"🛤️ A* path calculated: {path}")
            return path

        for neighbor in grid.get_neighbors(current):
            tentative_g_score = current_g + 1
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal_node)
                heapq.heappush(open_set, (f_score[neighbor], tentative_g_score, neighbor))
                print(f"🛤️ Added neighbor: ({neighbor.x}, {neighbor.z}), f_score={f_score[neighbor]:.2f}")
    print(f"🛤️ A* path calculation failed: no path from ({start_node.x}, {start_node.z}) to ({goal_node.x}, {goal_node.z})")
    return []