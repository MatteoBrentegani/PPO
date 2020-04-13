#
import sys
import yaml

from ppo_lossFunction import PPO
#from ppo_V2 import PPO
from gym_unity.envs import UnityEnv

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':

    with open("config.yaml", 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    STATE_SIZE = cfg['network_size']['state_size']
    ACTION_SIZE = cfg['network_size']['action_size']
    ACTION_SIZE2 = cfg['network_size']['action_size2']
    BATCH_SIZE = cfg['network_size']['batch_size']

    DISCOUNT_FACTOR = cfg['hyperparameters']['discount_factor']

    EPISODES = cfg['training']['episodes']
    TARGET_MODEL_UPDATE = cfg['training']['target_update']

    env_name = "ENV/tb3_toolkitv0.10"  # "UnityProject/Build/Turtlebot"
    print("ENV START")
    ENV = UnityEnv(env_name, worker_id=0)

    print("END ENV")
    PPO(ENV, BATCH_SIZE, TARGET_MODEL_UPDATE, EPISODES, STATE_SIZE, ACTION_SIZE, ACTION_SIZE2)
