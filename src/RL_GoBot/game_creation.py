import torch
import time
from RL_GoBot.MCTSearch import MCTS
from RL_GoBot import var
from gym_go import gogame


def sample_one_self_play_MCTS(net):
    with torch.no_grad():
        data_set = []
        state = gogame.init_state(var.BOARD_SIZE)
        root = MCTS(state, None, 1)
        while not gogame.game_ended(state) : 
            print("root", root)
            tmp = time.time()
            
            for _ in range(var.N_TREE_SEARCH) : 
                root.push_search()
            MCTS_policy = MCTS.tree_policy(root)
            data_set.append([state, MCTS_policy])
        
            next_root = root.best_next_node()
            next_state = gogame.next_state(state, next_root.action)
            root = next_root
            state = next_state
            print("\n", time.time() - tmp)

    reward = gogame.winning(state, var.KOMI)
    black_area, white_area = gogame.areas(state)
    print(black_area, white_area)
    # if data_set[0,0][2,0,0] == 1:   # if the first to play is white (nevers happen)
    #     reward = -reward
    for move in data_set:
        move.append(reward)
        reward = -reward

    return data_set
    