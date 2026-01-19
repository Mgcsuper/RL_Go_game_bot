import numpy as np
import torch
from RL_GoBot import var
from RL_GoBot.model import GoBot
from gym_go import gogame, govars

with torch.no_grad():
    net = GoBot()
    state0 = gogame.init_state(var.BOARD_SIZE)
    state1 = gogame.init_state(var.BOARD_SIZE)
    state1[govars.TURN_CHNL, :, :] = 1
    state1[0, 2, 2] = 1

    result = net(state0).result[0]
    
    sorted_indices_actions = sorted(range(len(result)), key=lambda i: result[i])    # sort from the most willing move to play until the most unwilling move, by the actual policy
    
    invalid_moves = gogame.invalid_moves(state0)
    for i in range(var.BOARD_SIZE**2+1) :
        action = sorted_indices_actions[i]
        if not invalid_moves[action] :  # check if it is a valide move
            break
    print(action)

    result = net(state1).result[0]
    
    sorted_indices_actions = sorted(range(len(result)), key=lambda i: result[i])    # sort from the most willing move to play until the most unwilling move, by the actual policy
    
    invalid_moves = gogame.invalid_moves(state1)
    for i in range(var.BOARD_SIZE**2+1) :
        action = sorted_indices_actions[i]
        if not invalid_moves[action] :  # check if it is a valide move
            break
    print(action)
