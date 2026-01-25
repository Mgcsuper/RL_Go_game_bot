import torch
import numpy as np


from gym_go import gogame

from RL_GoBot.model import GoBot
from RL_GoBot import var




def action2d(action) :
    return action // var.BOARD_SIZE, action % var.BOARD_SIZE 

def action1d(action) : 
    return action[0] * var.BOARD_SIZE + action[1] 


def roll_policy(state, net : GoBot, prt = False):    
    """
    return :
        the score of the outcome of the self played game seen by the first player to play
    """
    player = gogame.turn(state)
    game_ended = gogame.game_ended(state)

    game_turns = 0
    while not game_ended:
        if game_turns < var.MAX_TURNS :
            if torch.all(torch.tensor(state[3,:,:]) == 1):  # if there is no more legal moves, the only move is pass
                action = var.BOARD_SIZE**2      # pass move
            else :
                result = net(state)[0].result.numpy()

                invalid_moves = gogame.invalid_moves(state).astype(np.bool)
                result[invalid_moves] = -float("inf")

                action = np.argmax(result)
        else :
            action = var.BOARD_SIZE**2
        
        # if prt : print(action2d(action))
        # if prt : print(state)
        state = gogame.next_state(state, action)
        game_ended = gogame.game_ended(state)
        game_turns += 1

    score = gogame.winning(state, var.KOMI)  # from absolut black perspective
    if player :
        score = - score
    return score        # from the perspective of the player who where supposed to play at the starting state of the roll_policy
 
    
