import numpy as np
import torch
import queue
import time

from RL_GoBot import var
from RL_GoBot.model import GoBot
from RL_GoBot.batch_MCTSearch import MCTS, Node
from RL_GoBot.batch_rollout import Continuos_Rollout

from gym_go import gogame

def test_():
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


def test_queue():
    with torch.no_grad():
        net = GoBot()
        tree = MCTS(net, None, None)
        tree.net.eval()
        tree.task_queue = queue.Queue(8)
        roll_worker = Continuos_Rollout(tree)

        init_state = gogame.init_state(var.BOARD_SIZE)
        action1 = 0
        action2 = 1
        action3 = 2
        Node1 = Node(init_state, action1, None, 1)
        Node2 = Node(init_state, action2, None, 1)
        Node3 = Node(init_state, action3, None, 1)
        next_state1 = gogame.next_state(init_state, action1)
        next_state2 = gogame.next_state(init_state, action2)
        next_state3 = gogame.next_state(init_state, action3)
        init_state[[0,2],:,:] = 1
        Nodef = Node(init_state, 81, None, 1)

        tree.task_queue.put((Node1, next_state1))
        tree.task_queue.put((Node2, next_state2))

        tmp = time.time()
        roll_worker.load_task()
        print(time.time() - tmp)
        print(roll_worker.batchs_active)

        for i in range(5):
            roll_worker.batch_one_rollout()
            roll_worker.batch_game_ended = gogame.batch_game_ended(roll_worker.batch_states)
        
        print(roll_worker.batch_states[3])
        print(roll_worker.batchs_active)
        print(roll_worker.batch_game_move_count)

        print("__second__")
        tree.task_queue.put((Node3, next_state3))

        tmp = time.time()
        roll_worker.load_task()
        print(time.time() - tmp)
        print(roll_worker.batchs_active)

        for i in range(5):
            roll_worker.batch_one_rollout()
            roll_worker.batch_game_ended = gogame.batch_game_ended(roll_worker.batch_states)

        
        print(roll_worker.batch_states[3])
        print(roll_worker.batchs_active)
        print(roll_worker.batch_game_move_count)
        print(roll_worker.batch_game_ended)

        print("__third__")
        tree.task_queue.put((Nodef, init_state))

        tmp = time.time()
        roll_worker.load_task()
        print(time.time() - tmp)
        print(roll_worker.batchs_active)

        for i in range(3):
            print(f"___{i}___")
            roll_worker.batch_one_rollout()
            roll_worker.batch_game_ended = gogame.batch_game_ended(roll_worker.batch_states)
            print(roll_worker.batch_states[3])
            print(roll_worker.batch_game_ended)

        
        print(roll_worker.batchs_active)
        print(roll_worker.batch_game_move_count)



    roll_worker.batch_states = gogame.batch_init_state(var.ROLL_BATCH_SIZE, var.BOARD_SIZE)
    roll_worker.batchs_active = np.ones((var.ROLL_BATCH_SIZE - 1), dtype = bool)
    with torch.no_grad():
        tree.net.eval()
        for i in range(5):
            roll_worker.batch_one_rollout()
            print("\n", i, " on essay")
            print(roll_worker.batch_states[0])

