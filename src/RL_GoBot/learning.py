import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import math

from RL_GoBot import var 
from RL_GoBot.model import GoBot
from RL_GoBot.data_base import get_data




def compute_loss(output, target, reward : torch.Tensor):
  value_loss = F.mse_loss(output.value, reward)
  log_probs = F.log_softmax(output.result, dim=1)
  policy_loss = F.kl_div(log_probs, target, reduction='batchmean')
  # regulation = var.L2_LOSS
  return var.REGRESSION_LOSS * value_loss + var.CLASSIFICATION_LOSS * policy_loss 


def get_optimizer(net, lr, wd, momentum):
  optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
  return optimizer



def test(net : GoBot, data_loader : DataLoader, device='cuda:0'):
    with torch.no_grad():
      net.eval()
      cumulation_loss = 0
      for batch_idx, (state, targets, reward) in enumerate(data_loader):
          # Load data into GPU
          state = state.to(device)
          targets = targets.to(device)
          reward = reward.to(device)

          # Forward pass
          outputs = net(state)

          # Apply the loss
          cumulation_loss += compute_loss(outputs, targets, reward)
      return cumulation_loss
      


def train(net : GoBot, data_loader : DataLoader, optimizer : torch.optim.SGD, device='cuda:0', temperature=0):

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

    # applie some randomness in the grad
    for p in net.parameters():
      if p.grad is not None:
          noise = torch.randn_like(p.grad) * temperature
          p.grad += noise

    # Update parameters
    optimizer.step()

    

def train_one_episode(net : GoBot,
                      db : Dataset,
                      batch_size=var.BATCH_SIZE,
                      device='cuda:0',
                      learning_rate=var.LEARNING_RATE,
                      temperature=0,
                      weight_decay=var.L2_LOSS,
                      momentum=var.MOMENTUM,
                      epochs=var.EPOCHS):
      '''
      Input arguments
      batch_size: Size of a mini-batch
      device: GPU where you want to train your network
      weight_decay: Weight decay co-efficient for regularization of weights
      momentum: Momentum for SGD optimizer
      epochs: Number of epochs for training the network
      '''
  
      optimizer = get_optimizer(net, learning_rate, weight_decay, momentum)

      for e in range(epochs):
          print("epoch : ", e)
          train_loader = get_data(db, batch_size)
          train(net, train_loader, optimizer, device=device, temperature=temperature)