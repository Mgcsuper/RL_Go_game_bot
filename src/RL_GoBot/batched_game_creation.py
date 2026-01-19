import torch
import time

import threading
import queue

from RL_GoBot.batch_MCTSearch import MCTS, Node
from RL_GoBot.rollout import Continuos_Rollout
from RL_GoBot.model import GoBot
from RL_GoBot import var
from RL_GoBot.data_base import GoDatabase

from gym_go import gogame


def one_self_play_MCTS(net : GoBot):
    with torch.no_grad():
        data_set = []
        state = gogame.init_state(var.BOARD_SIZE)
        root_node = Node(state, None, None, 1, root = True)
        task_queue = queue.Queue(maxsize=var.MAX_QUEUE_SIZE)
        tree = MCTS(net, task_queue, root_node)
        roll_out_object = Continuos_Rollout(tree)

        while not gogame.game_ended(state) :             
            print("\n - new root - \n", tree)
            tmp = time.time()

            worker = threading.Thread(target=roll_out_object.continus_batch_roll_out)
            # print("start")
            worker.start()

            for i in range(var.N_TREE_SEARCH) : 
                tree.push_search(tree.root)

            tree.task_queue.put(None)
            worker.join()

            
            MCTS_policy = tree.tree_policy()
            data_set.append([state, MCTS_policy])
        
            print("- general tree time and process info -", flush=True)
            print("nomber of rollout : ", MCTS.roll_policy_count, flush=True)
            print("nomber of forward : ", GoBot.forward_count, flush=True)
            print("time for this move : ", time.time() - tmp, flush=True)

            next_root = tree.next_node()
            next_root.root = True
            next_state = gogame.next_state(state, next_root.action)
            tree.root = next_root
            state = next_state
            MCTS.roll_policy_count = 0
            GoBot.forward_count = 0
            # print("what")


    reward = gogame.winning(state, var.KOMI)
    for move in data_set:
        move.append(reward)
        reward = -reward

    return data_set


def self_play_MCTS(N, net : GoBot, db : GoDatabase):    # obliged when playing more then one game to save them in a db
    net.eval()
    for i in range(N):
        tmp_total = time.time()
        game_moves = one_self_play_MCTS(net)
        db.save_one_game(game_moves)
        
        print("     game -{}-".format(i))
        print("number of moves {}".format(len(game_moves)))
        print("duration of total game creation : {}".format(time.time() - tmp_total))
    return


    
from RL_GoBot.data_base import GoDatabase, get_data

if __name__ == "__main__":

    net = GoBot()
    one_self_play_MCTS(net)