from RL_GoBot.model import GoBot
from RL_GoBot import var
from RL_GoBot.MCTSearch import roll_policy, MCTS
from gym_go.envs import go_env
from gym_go import gogame

net = GoBot()

def test_net_forward():
    env = go_env.GoEnv(size=7, komi=0)
    state, reward, done, info = env.step((0,0))

    out = net.forward(state)
    print(out.result[0])
    print(out.result[0,10])

def test_roll_policy():
    state = gogame.init_state(var.BOARD_SIZE)
    print(roll_policy(state, net))


def test_concecutive_push_search():
    state = gogame.init_state(var.BOARD_SIZE)
    MCTS.net = net
    root = MCTS(state, None, 1)
    for i in range(var.N_THRESHOLD + 3) :
        root.push_search()


