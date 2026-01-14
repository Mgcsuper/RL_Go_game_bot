from RL_GoBot import var
from config import MODEL_DIR
import torch

print(var.BOARD_SIZE)

class OutputFormating(torch.Tensor):
    @property
    def value(self):
        """Return the last element along dim=-1"""
        if self.ndim == 1:
            return self[-1]
        else:
            return self[..., -1]

    @property
    def result(self):
        """Return all elements except the last along dim=-1"""
        if self.ndim == 1:
            return self[:-1]
        else:
            return self[..., :-1]


class GoBot(torch.nn.Module):
  """
  See only the bord as playing as the black player 
  (so when it's white we just exchange the black and white color so it can see it as black)
  """
  def __init__(self):
    super(GoBot, self).__init__()

    # input channel = 4, output channels = 20, kernel size = 7, padding = 3
    # input image size = (BOARDSIZE, BOARDSIZE), image output size = (BOARDSIZE, BOARDSIZE)
    self.convol1 = torch.nn.Conv2d(4, 20, 7, padding=3, bias=True)
    
    self.convol2 = torch.nn.Conv2d(20, 40, 5, padding=2, bias=True)

    self.linear1 = torch.nn.Linear(40*(var.BOARD_SIZE**2), 150, bias=True)

    # input dim = 120, output dim = 84
    self.linear2 = torch.nn.Linear(150, 110, bias=True) 

    self.linear3 = torch.nn.Linear(110, var.BOARD_SIZE**2 + 2, bias=True) # 83 pour les 81 case possible le suivant pass et le dernier pour la valeur


  def forward(self, x):
    ##--- reduse the 6 channel input of the environement that i have import to a 4 channel ---##
    if not isinstance(x, torch.Tensor):
      x = torch.tensor(x, dtype=torch.float32)

    if x.ndim == 3: 
        x = x.unsqueeze(0)

    if x[:,2,0,0] == 1:
      x[:, [0,1], ...] = x[:, [1,0], ...]
    
    x = x[:,[0,1,3,4],...]
    ##------##

    x = torch.nn.ReLU() (self.convol1(x))

    x = torch.nn.ReLU() (self.convol2(x))

    # flatten the feature maps into a long vector
    x = x.view(x.shape[0], -1) 

    x = torch.nn.ReLU() (self.linear1(x))
    x = torch.nn.ReLU() (self.linear2(x))
    x = self.linear3(x)

    return OutputFormating(x)
  

  def save_model(self, file_name : str):
     torch.save(self.state_dict(), MODEL_DIR/file_name)
     return 
  

  def load_model(self, file_name : str):
     self.load_state_dict(torch.load(MODEL_DIR/file_name))
     return