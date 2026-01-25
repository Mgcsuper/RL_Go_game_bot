import numpy as np
import torch 

from RL_GoBot.model import GoBot
from RL_GoBot import var

from gym_go import gogame


global_net : GoBot = None


def init_roll_policy(model):
    global global_net
    global_net = model
    

def roll_policy(state : np.ndarray):    
    """
    return :
        the score of the outcome of the self played game seen by the first player to play
    """
    player = state[2,0,0]
    game_ended = gogame.game_ended(state)

    game_turns = 0
    while not game_ended:
        if game_turns < var.MAX_TURNS :
            if torch.all(torch.tensor(state[3,:,:]) == 1):  # if there is no more legal mouves, the only move is pass
                action = var.BOARD_SIZE**2      # pass move
            else :
                result = global_net(state).result[0]
                # print(result)
                sorted_indices_actions = sorted(range(len(result)), key=lambda i: result[i])    # sort from the most willing move to play until the most unwilling move, by the actual policy
                invalid_moves = gogame.invalid_moves(state)
                for i in range(var.BOARD_SIZE**2+1) :
                    action = sorted_indices_actions[i]
                    if not invalid_moves[action] :  # check if it is a valide move
                        break
        else :
            action = var.BOARD_SIZE**2
        
        state = gogame.next_state(state, action)
        game_ended = gogame.game_ended(state)
        game_turns += 1

    count = GoBot.forward_count  
    GoBot.forward_count  = 0

    score = gogame.winning(state, var.KOMI)  # from absolut black perspective
    if player :
        score = - score
    
    return score, count       # from the perspective of the player who where supposed to play at the starting state of the roll_policy
 

if __name__ == "__main__":
    print("wooo rollout")
    # import multiprocessing
    # multiprocessing.freeze_support()