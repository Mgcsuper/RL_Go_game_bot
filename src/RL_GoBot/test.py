import torch
import numpy as np
from RL_GoBot.model import GoBot

net = GoBot()
net.eval

with torch.no_grad():
    batch_states = np.zeros((8, 6, 9, 9), dtype=np.float32)
    for i in range(8):
        batch_states[i] = np.random.rand(9, 9)

    batch_net_output = net(batch_states)
    batch_values = batch_net_output[:,-1]
    batch_net_policy = batch_net_output[:,:-1]
    print(batch_net_output.shape)
    print(batch_values)
    print(batch_net_policy.shape)

    sorted_indices = np.argsort(batch_net_policy, axis=1)

    print(sorted_indices)