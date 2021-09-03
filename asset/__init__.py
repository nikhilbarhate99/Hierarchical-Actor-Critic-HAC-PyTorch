from asset.continuous_mountain_car import Continuous_MountainCarEnv
from asset.pendulum import PendulumEnv

from gym.envs.registration import register

register(
    id="MountainCarContinuous-h-v1",
    entry_point="asset:Continuous_MountainCarEnv",
)

register(
    id="Pendulum-h-v1",
    entry_point="asset:PendulumEnv",
)
