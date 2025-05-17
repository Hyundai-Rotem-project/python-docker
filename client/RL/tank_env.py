import gym
from gym import spaces
import numpy as np

class TankEnv(gym.Env):
    def __init__(self, grid_size=300, pixel_size=(0.514, 0.76), max_steps=200):
        super(TankEnv, self).__init__()
        self.grid_size = grid_size
        self.pixel_size_x, self.pixel_size_z = pixel_size
        self.max_steps = max_steps

        self.obstacles = np.zeros((grid_size, grid_size), dtype=bool)
        self.action_space = spaces.Discrete(8)  # 8방향 이동
        self.observation_space = spaces.Box(low=0, high=1, shape=(grid_size, grid_size, 1), dtype=np.uint8)

        self.agent_pos = None
        self.goal_pos = None
        self.steps = 0

    def reset(self):
        self.steps = 0
        self.agent_pos = self._random_free_position()
        self.goal_pos = self._random_free_position()
        while self.goal_pos == self.agent_pos:
            self.goal_pos = self._random_free_position()
        return self._get_obs()

    def step(self, action):
        dx, dz = self._action_to_delta(action)
        new_x = np.clip(self.agent_pos[0] + dx, 0, self.grid_size - 1)
        new_z = np.clip(self.agent_pos[1] + dz, 0, self.grid_size - 1)

        if not self.obstacles[new_x, new_z]:
            self.agent_pos = (new_x, new_z)

        self.steps += 1
        done = self.agent_pos == self.goal_pos or self.steps >= self.max_steps
        reward = 1.0 if self.agent_pos == self.goal_pos else -0.01
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        obs = np.zeros((self.grid_size, self.grid_size, 1), dtype=np.uint8)
        obs[self.agent_pos[0], self.agent_pos[1], 0] = 128
        obs[self.goal_pos[0], self.goal_pos[1], 0] = 255
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
            gx0 = int((obs["x_min"] - x_min) * scale_x)
            gx1 = int((obs["x_max"] - x_min) * scale_x)
            gz0 = int((obs["z_min"] - z_min) * scale_z)
            gz1 = int((obs["z_max"] - z_min) * scale_z)
            self.obstacles[gx0:gx1+1, gz0:gz1+1] = True
