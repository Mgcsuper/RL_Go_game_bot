import argparse
from gym_go.envs_copy import go_env

# Arguments
parser = argparse.ArgumentParser(description='Demo Go Environment')
parser.add_argument('--boardsize', type=int, default=7)
parser.add_argument('--komi', type=float, default=0)
args = parser.parse_args()

# Initialize environment
env = go_env.GoEnv(size=args.boardsize, komi=args.komi)

# Game loop
done = False
while not done:
    action = env.render(mode = 'human')
    print(action)
    state, reward, done, info = env.step(action)
    print(state, reward, done, info)

    if env.game_ended():
        break
    action = env.uniform_random_action()
    state, reward, done, info = env.step(action)
env.render()
