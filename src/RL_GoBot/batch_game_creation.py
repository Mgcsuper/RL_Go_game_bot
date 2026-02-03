import torch
import time

import threading
import queue
 
from RL_GoBot.batch_MCTSearch import MCTS
from RL_GoBot.batch_rollout import Continuos_Rollout
from RL_GoBot.Node import Node
from RL_GoBot.model import GoBot
from RL_GoBot.data_base import GoDatabaseMongo
from RL_GoBot import var

from gym_go import gogame



class ThreadWithException(threading.Thread):
    def run(self):
        self.exc = None
        try:
            if self._target:
                self._target(*self._args, **self._kwargs)
        except Exception as e:
            self.exc = e


def one_move_timed(tree:MCTS, roll_out_object:Continuos_Rollout, reflexion_time = 0, min_push_search = var.N_THRESHOLD + 1):
    """
    modify directly the MCTS tree object, with who you will use tree.policy() or tree.next_node()
    """
    # creation of the worker Thread
    worker = ThreadWithException(target=roll_out_object.continus_batch_roll_out)
    worker.start()

    # N_TREE_SEARCH times push_search()
    try :
        tmp = time.time()
        i = 0
        while (time.time() - tmp < reflexion_time) or (i < min_push_search): 
            i+=1
            tree.push_search(tree.root)
            # if there is a raise Excepetion in the child thread
            if worker.exc :
                raise worker.exc 
        # print(time.time() - tmp)
        # print(i)
    except KeyboardInterrupt :
        tree.task_queue.put(None)
        worker.join()
        raise
    except :
        raise

    # finish the push_search(), end of the tree extention
    tree.task_queue.put(None)
    worker.join()

    # return the policy of this tree
    return 


def one_move_counted(tree:MCTS, roll_out_object:Continuos_Rollout) -> torch.tensor:
    # creation of the worker Thread
    worker = ThreadWithException(target=roll_out_object.continus_batch_roll_out)
    worker.start()

    # N_TREE_SEARCH times push_search()
    try :
        for _ in range(var.N_TREE_SEARCH) : 
            tree.push_search(tree.root)
            # if there is a raise in the child thread
            if worker.exc :
                raise worker.exc 
    except KeyboardInterrupt :
        tree.task_queue.put(None)
        worker.join()
        raise
    except :
        raise

    # finish the push_search(), end of the tree extention
    tree.task_queue.put(None)
    worker.join()

    return 


def one_game(tree:MCTS, roll_out_object:Continuos_Rollout, state) :
    data_set = []   # list of move, a move being a list of 3 tensor : (state, policy, reward)
    moves_count = 0
    while not gogame.game_ended(state) and moves_count < var.MAX_TURNS:  
        # print info
        print("\n - new root - \n", tree)
        tmp = time.time()
        #

        # one_move tree creation
        one_move_counted(tree, roll_out_object)

        MCTS_policy = torch.tensor(tree.tree_policy(), dtype=torch.float32)
        tensor_state = torch.tensor(state, dtype=torch.float32)
        data_set.append([tensor_state, MCTS_policy])

        # update of the tree root, the curent state, 
        next_root = tree.next_node()
        if next_root is None :
            return data_set, state, True
        next_root.is_root = True
        tree.root = next_root
        next_state = gogame.next_state(state, next_root.action)
        state = next_state
        moves_count += 1

        # print info
        print("- general tree time and process info -", flush=True)
        print("nomber of rollout : ", tree.roll_policy_count, flush=True)
        print("nomber of forward : real {} | equivalent roll {}, and extend {}".format(GoBot.forward_count, tree.roll_forward_count, tree.extend_count), flush=True)
        print("time for this move : ", time.time() - tmp, flush=True)
        #

        # debug and statistic variables
        tree.roll_policy_count = 0
        tree.roll_forward_count = 0
        tree.extend_count = 0
        GoBot.forward_count = 0
        ##
    
    return data_set, state, False



def one_self_play_MCTS(net : GoBot, temperature = var.TEMPERATURE_MCTS):
    # initialize most of the necessery objetcs, and creat a data_set for a game
    with torch.no_grad():
        net.eval()
        data_set : list
        initial_state = gogame.init_state(var.BOARD_SIZE)
        root_node = Node(initial_state, None, 1, parent=None, depth=0, root=True)
        task_queue = queue.Queue(maxsize=var.MAX_QUEUE_SIZE)
        tree = MCTS(net, root_node, task_queue, temperature)
        roll_out_object = Continuos_Rollout(tree)

        data_set, end_state, resigne = one_game(tree, roll_out_object, initial_state)

    # back propagate the reward of this game
    if resigne :
        reward = 1.0 if gogame.turn(end_state) else -1.0
    else :
        reward = float(gogame.winning(end_state, var.KOMI))
    reward = torch.tensor(reward, dtype=torch.float32)

    for move in data_set:
        move.append(reward)
        reward = -reward

    # return the data_set that will be stored in the data base 
    return data_set


def self_play_MCTS(N, net : GoBot, db : GoDatabaseMongo):    # obliged when playing more then one game to save them in a db
    for i in range(N):
        tmp_total = time.time()
        game_moves = one_self_play_MCTS(net)
        
        ## print info
        print("     game -{}-".format(i))
        print("number of moves {}".format(len(game_moves)))
        print("duration of total game creation : {}".format(time.time() - tmp_total))
        ##

        # save data in the database
        db.save_one_game(game_moves)
    return



if __name__ == "__main__":
    from config import DEVICE
    DEVICE = 'cpu' # 'cuda'
    var.N_TREE_SEARCH = 30
    net = GoBot()
    data = one_self_play_MCTS(net)
    # for i in range(len(data)) :
    #     print(f"__{i}__")
    #     print(gogame.str(data[i][0].numpy()))
    #     print(data[i][2])