import torch
from RL_GoBot import var 

def compute_loss(output, target, reward):
  value_loss = output.value - reward
  policy_loss = torch.dot(target, output.result)
  regulation = var.L2_LOSS
  return value_loss + policy_loss + regulation


def get_optimizer(net, lr, wd, momentum):
  optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
  return optimizer


def test(net, data_loader, cost_function, device='cuda:0'):
  samples = 0.
  cumulative_loss = 0.
  cumulative_accuracy = 0.

  net.eval() # Strictly needed if network contains layers which has different behaviours between train and test
  with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(data_loader):
      # Load data into GPU
      inputs = inputs.to(device)
      targets = targets.to(device)

      # Forward pass
      outputs = net(inputs)

      # Apply the loss
      loss = cost_function(outputs, targets)

      # Better print something
      samples+=inputs.shape[0]
      cumulative_loss += loss.item() # Note: the .item() is needed to extract scalars from tensors
      _, predicted = outputs.max(1)
      cumulative_accuracy += predicted.eq(targets).sum().item()

  return cumulative_loss/samples, cumulative_accuracy/samples*100


def train(net, data_loader, optimizer, device='cuda:0'):
  samples = 0.
  cumulative_loss = 0.


  net.train() # Strictly needed if network contains layers which has different behaviours between train and test
  for batch_idx, (state, targets, reward) in enumerate(data_loader):
    # Load data into GPU
    state = state.to(device)
    targets = targets.to(device)

    # Forward pass
    outputs = net(state)

    # Apply the loss
    loss = compute_loss(outputs, targets, reward)

    # Reset the optimizer

    # Backward pass
    loss.backward()

    # Update parameters
    optimizer.step()

    optimizer.zero_grad()

    # Better print something, no?
    samples+=state.shape[0]
    cumulative_loss += loss.item()
    # _, predicted = outputs[:82].max(1)   # for the action which has been remembered

  return cumulative_loss/samples