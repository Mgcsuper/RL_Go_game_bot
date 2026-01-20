import torch
from typing import List
import random
from math import sqrt
import numpy as np


import queue

from RL_GoBot import var
from RL_GoBot.model import GoBot

from gym_go import govars, gogame




class Node:
    def __init__(self, state, action: int, parent: "Node", p: torch.Tensor, depth=0, root=False):
        self.front_push = 0
        self.back_probpagation = 0

        self.root = root
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
            return "None"
        return self.action // var.BOARD_SIZE, self.action % var.BOARD_SIZE


    def update_Q(self):
        # Calcul de Q à partir de Wv et Wr
        if self.N > 0:
            self.Q = ((1 - var.QU_RATIO) * self.Wv + var.QU_RATIO * self.Wr) / self.N

    def __str__(self):
        print("___ node info ___")
        print("action : {} | depth : {}".format(self.action_2d(), self.depth))
        return "N : {}, P : {} \n".format(self.N, self.p)


class MCTS:
    roll_policy_count = 0
    def __init__(self, net: GoBot, task_queue : queue.Queue, root_node: Node, temperature = var.TEMPERATURE_MCTS):
        self.task_queue = task_queue
        self.root = root_node
        self.net = net
        self.temperature = temperature 
        self.root_depth = 0
        self.policy = None


    def extend_node(self, node : Node):
        # print("extend")
        next_state = node.next_state()

        # Prédiction du NN
        result = self.net(next_state).result[0]

        # Mask des coups invalides et softmax
        invalid_moves = gogame.invalid_moves(next_state)  # contain also the pass move
        result[invalid_moves == 1] = float('-inf')
        prior_probs = torch.softmax(result, dim=-1)

        for action, p in enumerate(prior_probs):
            if p != 0 : 
                node.next_nodes.append(Node(
                    next_state,
                    action,
                    node,
                    p,
                    node.depth + 1
                ))


    def back_prob(self, node: Node, value, roll):
        # print("back \n", node)
        node.back_probpagation += 1

        node.N += -var.VIRTUAL_LOSS + 1
        node.Wr += 3 + roll
        node.Wv += value
        # node.Q = ((1 - var.QU_RATIO) * node.Wv + var.QU_RATIO * node.Wr) /node.N
        node.update_Q()

        if not node.root:       # we dont make the root Node parametters evolve 
            self.back_prob(node.parent, -value, -roll)

        return


    def back_prob_leaf(self, node: Node, value, roll, count : int):
        # print("count : ", count)

        GoBot.forward_count += count
        node.roll = roll
        node.value = value
        self.back_prob(node, value, roll)


    def push_search(self, node: Node):
        # print("___")
        # print("push \n", node)
        node.front_push += 1

        node.N += var.VIRTUAL_LOSS
        node.Wr -= 3

        # Cas de la première exploration
        if node.first_search:
            # print("first")
            node.first_search = False
            next_state = node.next_state()
            try:
                self.task_queue.put((node, next_state))    # node.value = self.net(next_state).value  and   node.roll = roll_policy(next_state)
                MCTS.roll_policy_count += 1
                # print("put  ")
            except queue.Full:
                pass
            

        # Cas N >= threshold
        elif node.N >= var.N_THRESHOLD and not node.end_game:
            # print("> N")
            if not node.next_nodes:
                self.extend_node(node)

            # Sélection du next_node 
            node_to_search = max(
                node.next_nodes,
                key=lambda n: n.Q + var.C_PUCT * n.p * sqrt(node.N) / (1 + n.N)
            )
            self.push_search(node_to_search)

        # Cas 1 <= N < threshold
        else:
            # print("midel")
            self.back_prob(node, node.value, node.roll)
        # print("_")
        return



    def best_next_node(self):
        """Retourne l'enfant avec le plus grand nombre de visites N"""
        if not self.root.next_nodes:
            print("Error : No policy is possible without expending this node")
            return None
        best_node = max(self.root.next_nodes, key=lambda n: n.N)
        return best_node
    

    def next_node(self):
        """Échantillonne un enfant selon la policy"""
        if self.policy is None:
            self.tree_policy()
            
        r = random.random()
        self.affiche_policy()
        print("r : ", r, flush=True)
        cumulative = 0
        for node in self.root.next_nodes:
            cumulative += self.policy[node.action]
            if cumulative > r:
                print("action : ", node.action_2d(), flush=True)
                self.policy = None
                return node

        self.policy = None
        # Au cas où la somme n'est pas exactement 1 à cause des flottants
        return self.next_nodes[-1]
    

    def tree_policy(self):
        visited_N = [0 for i in range(var.BOARD_SIZE**2 + 1)]
        count_front = [0 for i in range(var.BOARD_SIZE**2 + 1)]
        count_back = [0 for i in range(var.BOARD_SIZE**2 + 1)]
        for node in self.root.next_nodes:
            visited_N[node.action] = node.N
            count_front[node.action] = node.front_push
            count_back[node.action] = node.back_probpagation

        self.count_branch_visit = visited_N  # juste pour de l'affichage
        self.count_front = count_front
        self.count_back = count_back

        visited_N_temperated = [N**(1 / self.temperature) for N in visited_N]
        diviseur = sum(visited_N_temperated)
        self.policy = [N / diviseur for N in visited_N_temperated]
        return self.policy


    def affiche_policy(self):
        tensor = torch.tensor(self.count_branch_visit[:-1])      # tensor 1D
        front = torch.tensor(self.count_front[:-1])      # tensor 1D
        back = torch.tensor(self.count_back[:-1])      # tensor 1D
        print("count mismatch : ", var.BOARD_SIZE**2 + 1 - np.count_nonzero(np.array(self.count_front) == np.array(self.count_back)))

        tensor_2d = tensor.view(var.BOARD_SIZE, var.BOARD_SIZE)
        front2d = front.view(var.BOARD_SIZE, var.BOARD_SIZE)
        back2d = back.view(var.BOARD_SIZE, var.BOARD_SIZE)

        print("- tree policy info -", flush=True)
        print("policy : ", tensor_2d, flush=True)
        # print("front :", front2d, flush=True)
        # print("back :", back2d, flush=True)
        print("pass : ", self.count_branch_visit[-1], flush=True)

    def __str__(self):
        return self.root.__str__()

