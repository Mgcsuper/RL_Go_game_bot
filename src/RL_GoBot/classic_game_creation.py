import torch
import time

from RL_GoBot.classic_MCTSearch import MCTS
from RL_GoBot.Node import Node
from RL_GoBot.model import GoBot
from RL_GoBot.data_base import GoDatabaseMongo
from RL_GoBot import var

from gym_go import gogame



def one_self_play_MCTS(net):
    with torch.no_grad():
        data_set = []
        state = gogame.init_state(var.BOARD_SIZE)
        root_node = Node(state, None, 1)
        tree = MCTS(net, root_node)
        while not gogame.game_ended(state) : 
            
            print("\n - new root - \n", tree)
            tmp = time.time()
            
            for _ in range(var.N_TREE_SEARCH) : 
                tree.push_search(tree.root)
            MCTS_policy = tree.tree_policy()
            data_set.append([state, MCTS_policy])
        
            print("- general tree time and process info -")
            print("nomber of rollout : ", MCTS.roll_policy_count)
            print("nomber of forward : ", GoBot.forward_count)
            print("time for this move : ", time.time() - tmp)

            next_root = tree.next_node()
            next_state = gogame.next_state(state, next_root.action)
            tree.root = next_root
            state = next_state
            MCTS.roll_policy_count = 0
            GoBot.forward_count = 0

    reward = gogame.winning(state, var.KOMI)
    for move in data_set:
        move.append(reward)
        reward = -reward

    return data_set


def self_play_MCTS(N, net, db : GoDatabaseMongo):    # obliged when playing mor then one game to save them somewhere
    for i in range(N):
        tmp_total = time.time()
        game_moves = one_self_play_MCTS(net)
        db.save_one_game(game_moves)
        
        print("     game -{}-".format(i))
        print("number of moves {}".format(len(game_moves)))
        print("duration of total game creation : {}".format(time.time() - tmp_total))


    