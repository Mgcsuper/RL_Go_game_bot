import torch
import time
from multiprocessing import Pool, Semaphore

from RL_GoBot.parallel_rollout import init_roll_policy
from RL_GoBot.parallel_MCTSearch import MCTS
from RL_GoBot.Node import Node
from RL_GoBot.model import GoBot
from RL_GoBot.data_base import GoDatabaseMongo
from RL_GoBot import var

from gym_go import gogame



def one_self_play_MCTS(net : GoBot):
    with torch.no_grad():
        data_set = []
        state = gogame.init_state(var.BOARD_SIZE)
        root_node = Node(state, None, 1, parent=None, depth=0, root=True)
        tree = MCTS(net, root_node)
        while not gogame.game_ended(state) : 
            
            print("\n - new root - \n", tree)
            tmp = time.time()
             
            tree.sem = Semaphore(var.MAX_PROCESSES)
            with Pool(processes=var.MAX_PROCESSES, initializer=init_roll_policy, initargs=(net,)) as pool:
                for i in range(var.N_TREE_SEARCH) : 
                    tree.sem.acquire()
                    tree.push_search(tree.root, pool)

                pool.close()
                pool.join()
            
            MCTS_policy = torch.tensor(tree.tree_policy(), dtype=torch.float32)
            tensor_state = torch.tensor(state, dtype=torch.float32)
            data_set.append([tensor_state, MCTS_policy])
        
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

    reward = torch.tensor(gogame.winning(state, var.KOMI), dtype=torch.float32)
    for move in data_set:
        move.append(reward)
        reward = -reward

    return data_set


def self_play_MCTS(N, net : GoBot, db : GoDatabaseMongo):    # obliged when playing more then one game to save them in a db
    net.eval()
    for i in range(N):
        tmp_total = time.time()
        game_moves = one_self_play_MCTS(net)
        db.save_one_game(game_moves)
        
        print("     game -{}-".format(i))
        print("number of moves {}".format(len(game_moves)))
        print("duration of total game creation : {}".format(time.time() - tmp_total))
    return



if __name__ == "__main__":
    print("wooo game creation")
    # import multiprocessing
    # multiprocessing.freeze_support()