import pytest
from gym_go.envs import GoEnv
from gym_go.envs import go_env


def test_go_env():
    env = go_env.GoEnv(size=7, komi=0)
    state, reward, done, info = env.step((0,0))
    print(state[:4], '\n----\n')
    state, reward, done, info = env.step((1,0))
    print(state[:4])

