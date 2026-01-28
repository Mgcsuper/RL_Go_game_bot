from RL_GoBot import var
import torch
import numpy as np
import os


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
    forward_count = 0
    def __init__(self):
        super(GoBot, self).__init__()

        # input channel = 4, output channels = 30, kernel size = 5, padding = 2
        # input image size = (BOARDSIZE, BOARDSIZE), image output size = (BOARDSIZE, BOARDSIZE)
        self.convol1 = torch.nn.Conv2d(4, 30, 5, padding=2, bias=True)
        
        self.convol2 = torch.nn.Conv2d(30, 40, 5, padding=2, bias=True)

        self.linear1 = torch.nn.Linear(40*int(((var.BOARD_SIZE-1)/2)**2), 150, bias=True)

        # input dim = 120, output dim = 84
        self.linear2 = torch.nn.Linear(150, 110, bias=True) 

        self.linear3 = torch.nn.Linear(110, var.BOARD_SIZE**2 + 2, bias=True) # 83 pour les 81 case possible le suivant pass et le dernier pour la valeur


    def forward(self, entry : torch.Tensor) -> OutputFormating:
        ##--- reduse the 6 channel input of the environement that i have import to a 4 channel ---##
        assert isinstance(entry, torch.Tensor), "Error: input must be a torch.Tensor"

        if entry.ndim == 3: 
            entry = entry.unsqueeze(0)

        x = entry.clone()
        batch_size = x.shape[0]


        for batch_idx in range(batch_size):
            if x[batch_idx,2,0,0] == 1:
                x[batch_idx, [0,1], ...] = x[batch_idx, [1,0], ...]
        
        x = x[:,[0,1,3,4],...].float()
        ##------##
        
        GoBot.forward_count += 1

        x = torch.nn.ReLU() (self.convol1(x))

        x = torch.nn.ReLU() (torch.nn.functional.max_pool2d(self.convol2(x), 3, 2))
        

        # flatten the feature maps into a long vector
        x = x.view(x.shape[0], -1) 

        x = torch.nn.ReLU() (self.linear1(x))
        x = torch.nn.ReLU() (self.linear2(x))
        x = self.linear3(x)

        return OutputFormating(x)
    

    def save_model(self, file_name : str):
        """
        file_name: the complite file_name (for example MODEL_9X9/TYPE)
        """
        torch.save(self.state_dict(), file_name)
        return 
    

    def load_model(self, file_name : str):    
        """
        file_name: the complite file_name (for example MODEL_9X9/TYPE)
        """
        path = file_name
        if os.path.exists(path):
            self.load_state_dict(torch.load(path))
        else :
            self.save_model(file_name)
        return
  

if __name__ == "__main__":
    model = GoBot()
    for name, param in model.named_parameters():
        print(name, param.requires_grad)