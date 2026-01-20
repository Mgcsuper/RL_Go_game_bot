import numpy as np
import torch

from RL_GoBot import var
from RL_GoBot.model import GoBot
from RL_GoBot.batch_MCTSearch import MCTS
from RL_GoBot.rollout import Continuos_Rollout

from gym_go import gogame

def test_():
    import queue

    net = GoBot()
    tree = MCTS(net, None, None)
    roll_worker = Continuos_Rollout(tree)

    roll_worker.batch_states = gogame.batch_init_state(var.ROLL_BATCH_SIZE, var.BOARD_SIZE)
    roll_worker.batchs_active = np.ones((var.ROLL_BATCH_SIZE), dtype = bool)
    with torch.no_grad():
        tree.net.eval()
        for i in range(10):
            roll_worker.batch_one_rollout()
            print("\n", i, " on essay")
            print(roll_worker.batch_states[0])