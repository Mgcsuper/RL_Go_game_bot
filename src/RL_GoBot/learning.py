import torch
from RL_GoBot import var 
from RL_GoBot.model import GoBot, OutputFormating
from torch.utils.data import DataLoader
import torch.nn.functional as F



def compute_loss(output, target, reward : torch.Tensor):
  value_loss = F.mse_loss(output.value, reward)
  log_probs = F.log_softmax(output.result, dim=1)
  policy_loss = F.kl_div(log_probs, target, reduction='batchmean')
  # regulation = var.L2_LOSS
  return var.REGRESSION_LOSS * value_loss + var.CLASSIFICATION_LOSS * policy_loss 


def get_optimizer(net, lr, wd, momentum):
  optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
  return optimizer



def train(net : GoBot, data_loader : DataLoader, optimizer : torch.optim.SGD, device='cuda:0'):
  samples = 0.
  cumulative_loss = 0.


  net.train() # Strictly needed if network contains layers which has different behaviours between train and test
  for batch_idx, (state, targets, reward) in enumerate(data_loader):
    # Load data into GPU
    state = state.to(device)
    targets = targets.to(device)
    reward = reward.to(device)

    # Forward pass
    outputs = net(state)

    # Apply the loss
    loss = compute_loss(outputs, targets, reward)

    optimizer.zero_grad()

    # Backward pass
    loss.backward()

    # Update parameters
    optimizer.step()


  #   # Better print something, no?
  #   samples+=state.shape[0]
  #   cumulative_loss += loss.item()
  #   # _, predicted = outputs[:82].max(1)   # for the action which has been remembered

  # return cumulative_loss/samples