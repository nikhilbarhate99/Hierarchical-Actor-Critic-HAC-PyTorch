from asset.continuous_mountain_car import Continuous_MountainCarEnv
from asset.pendulum import PendulumEnv

from gym.envs.registration import register

register(
    id="MountainCarContinuous-h-v1",
    entry_point="asset:Continuous_MountainCarEnv",
    max_episode_steps=50,
)

register(
    id="Pendulum-h-v1",
    entry_point="asset:PendulumEnv",
    max_episode_steps=50,
)
