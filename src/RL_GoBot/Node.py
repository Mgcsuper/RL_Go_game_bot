
import torch

from typing import List

from gym_go import gogame, govars

from RL_GoBot import var




class Node:
    def __init__(self, state, action: int, p: torch.Tensor, parent: "Node" = None, depth=0, root=False):
        self.front_push = 0
        self.back_propagation = 0

        self.is_root = root
        self.parent = parent
        self.state = state
        self.action = action
        self.p = p
        self.depth = depth
        self.end_game = self.next_state()[govars.DONE_CHNL,0,0]

        self.N = 0
        self.Wv = 0
        self.Wr = 0
        self.Q = 0

        self.next_nodes : List[Node] = []

        self.first_search = True
        self.value = 0
        self.roll = 0

    def next_state(self):
        if self.action is None:
            return self.state
        return gogame.next_state(self.state, self.action)

    def action_2d(self):
        if self.action is None:
            return None
        return self.action // var.BOARD_SIZE, self.action % var.BOARD_SIZE


    def update_Q(self):
        # Calcul de Q à partir de Wv et Wr
        if self.N > 0:
            self.Q = ((1 - var.QU_RATIO) * self.Wv + var.QU_RATIO * self.Wr) / self.N

    def __str__(self):
        print("___ node info ___")
        if gogame.turn(self.state) == 1:
            print("Black after action taken of this node")
        else :
            print("white after action taken of this node")

        print("action : {} | depth : {}".format(self.action_2d(), self.depth))
        return "N : {}, P : {} \n".format(self.N, self.p)


if __name__ == "__main__":
    print("tu fais quoi ici")
    # import multiprocessing
    # multiprocessing.freeze_support()