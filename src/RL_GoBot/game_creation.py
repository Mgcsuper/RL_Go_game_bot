import torch
import time
from RL_GoBot.MCTSearch import MCTS, Node
from RL_GoBot import var
from RL_GoBot.data_base import GoDatabase
from gym_go import gogame



def one_self_play_MCTS(net):
    with torch.no_grad():
        data_set = []
        state = gogame.init_state(var.BOARD_SIZE)
        root_node = Node(state, None, 1)
        tree = MCTS(net, root_node)
        while not gogame.game_ended(state) : 
            print("new root \n", tree)
            tmp = time.time()
            
            for _ in range(var.N_TREE_SEARCH) : 
                tree.push_search(tree.root)
            MCTS_policy = tree.tree_policy()
            data_set.append([state, MCTS_policy])
        
            next_root = tree.next_node()
            next_state = gogame.next_state(state, next_root.action)
            tree.root = next_root
            state = next_state
            print("\n", time.time() - tmp)

    reward = gogame.winning(state, var.KOMI)
    # black_area, white_area = gogame.areas(state)
    # if data_set[0,0][2,0,0] == 1:   # if the first to play is white (nevers happen)
    #     reward = -reward
    for move in data_set:
        move.append(reward)
        reward = -reward

    return data_set


def self_play_MCTS(N, net, db : GoDatabase):
    # data = []
    for i in range(N):
        tmp_total = time.time()
        game_moves = one_self_play_MCTS(net)
        db.save_one_game(game_moves)
        
        print("     game -{}-".format(i))
        print("number of moves {}".format(len(game_moves)))
        print("duration of total game creation : {}".format(time.time() - tmp_total))

        # data.extend(game_moves)
    # return data

    