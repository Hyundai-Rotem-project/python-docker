from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from tank_env import TankEnv
from gym.envs.registration import register
import get_obstacles_point as get_obstacles

# 1. 장애물 로드
obstacle_list = get_obstacles.load_obstacles('./test_April3.map')

# 2. 환경 등록
register(
    id='TankAvoid-v0',                      # 사용할 Gym 환경 이름
    entry_point='tank_env:TankAvoidEnv',   # '파일명:클래스이름' 형식
)

# 3. 환경 생성 및 장애물 세팅
env = TankEnv()
env.set_obstacles_from_list(obstacle_list, x_range=300, z_range=300, x_min=0, z_min=0)
check_env(env)

# 4. PPO 모델 생성 및 학습
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_tank_pathfinder_tensorboard/")
model.learn(total_timesteps=100_000) # 학습할 총 스텝 수 ; 10만 스텝

# 저장
model.save("ppo_tank_pathfinder")