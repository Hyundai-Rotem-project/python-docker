import gym
from gymnasium import spaces
import numpy as np

class TankEnv(gym.Env):
    def __init__(self, grid_size=300, pixel_size=(0.514, 0.76), max_steps=200):
        super(TankEnv, self).__init__()
        self.grid_size = grid_size
        self.pixel_size_x, self.pixel_size_z = pixel_size
        self.max_steps = max_steps
        self.episode_rewards = []
        self.episode_steps = []
        self.success_count = 0

        self.obstacles = np.zeros((grid_size, grid_size), dtype=bool)
        self.action_space = spaces.Discrete(8)  # 8방향 이동
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(grid_size, grid_size, 1), dtype=np.float32)

        self.agent_pos = None
        self.goal_pos = None
        self.steps = 0

    def reset(self):
        self.steps = 0
        self.episode_reward = 0.0
        self.agent_pos = self._random_free_position()
        self.goal_pos = self._random_free_position()
        while self.goal_pos == self.agent_pos:
            self.goal_pos = self._random_free_position()
        
        self.visited = set()
        self.visited.add(self.agent_pos)  # 초기 위치 방문 기록
        
        return self._get_obs()

    def step(self, action):
        dx, dz = self._action_to_delta(action)
        new_x = np.clip(self.agent_pos[0] + dx, 0, self.grid_size - 1)
        new_z = np.clip(self.agent_pos[1] + dz, 0, self.grid_size - 1)
        self.episode_reward += reward

        # 초기 거리 계산
        old_dist = np.linalg.norm(np.array(self.goal_pos) - np.array(self.agent_pos))
        
        # 이동 시도
        moved = False
        if not self.obstacles[new_x, new_z]:
            self.agent_pos = (new_x, new_z)
            moved = True

        # 새 거리 계산
        new_dist = np.linalg.norm(np.array(self.goal_pos) - np.array(self.agent_pos))

        # 중복 방문 감지
        if self.agent_pos in self.visited:
            revisit_penalty = -0.2
        else:
            self.visited.add(self.agent_pos)
            revisit_penalty = 0.0
        
        # 보상 계산
        reward = 0.0
        if self.agent_pos == self.goal_pos:
            reward = 10.0  # 도착 보상
        elif not moved:
            reward = -1.0  # 장애물에 부딪힘
        else:
            distance_reward = (old_dist - new_dist) * 0.1
            step_penalty = -0.01  # 이동할 때마다 누적 패널티
            reward = distance_reward + revisit_penalty + step_penalty
        
        reward = 1.0 if self.agent_pos == self.goal_pos else -0.01

        self.steps += 1
        done = self.agent_pos == self.goal_pos or self.steps >= self.max_steps

        if done:
            self.episode_rewards.append(self.episode_reward)
            self.episode_steps.append(self.steps)
            if self.agent_pos == self.goal_pos:
                self.success_count += 1
            print(f"[Episode Done] reward: {self.episode_reward:.2f}, steps: {self.steps}, success: {self.agent_pos == self.goal_pos}")
        
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        obs = np.zeros((self.grid_size, self.grid_size, 1), dtype=np.uint8)
        obs[self.agent_pos[0], self.agent_pos[1], 0] = 0.5
        obs[self.goal_pos[0], self.goal_pos[1], 0] = 1.0
        return obs

    def _action_to_delta(self, action):
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]
        return deltas[action]

    def _random_free_position(self):
        while True:
            x = np.random.randint(0, self.grid_size)
            z = np.random.randint(0, self.grid_size)
            if not self.obstacles[x, z]:
                return (x, z)

    def set_obstacles_from_list(self, obstacle_list, x_range, z_range, x_min, z_min):
        scale_x = self.grid_size / x_range
        scale_z = self.grid_size / z_range
        for obs in obstacle_list:
            gx =  int((obs[0] - x_min) * scale_x)
            gz =  int((obs[1] - z_min) * scale_z)
            if 0 <= gx < self.grid_size and 0 <= gz < self.grid_size:
                self.obstacles[gx, gz] = True
