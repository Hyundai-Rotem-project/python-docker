from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from tank_env import TankNavigationEnv

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from tank_env import TankNavigationEnv
from gym.envs.registration import register

register(
    id='TankAvoid-v0',                      # 사용할 Gym 환경 이름
    entry_point='tank_env:TankAvoidEnv',   # '파일명:클래스이름' 형식
)
env = TankNavigationEnv()
check_env(env)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000) # 학습할 총 스텝 수 ; 10만 스텝

# 저장
model.save("ppo_tank_pathfinder")