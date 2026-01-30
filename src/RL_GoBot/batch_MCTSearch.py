import torch
import random
from math import sqrt
import numpy as np

import queue

from RL_GoBot import var
from RL_GoBot.model import GoBot
from RL_GoBot.Node import Node

from gym_go import gogame

from config import DEVICE



class MCTS:
    def __init__(self, net: GoBot, root_node: Node, task_queue : queue.Queue, temperature = var.TEMPERATURE_MCTS):
        # for statistic
        self.roll_policy_count = 0
        self.roll_forward_count = 0
        self.extend_count = 0

        self.task_queue = task_queue
        self.root = root_node
        self.net = net
        self.temperature = temperature 
        self.root_depth = 0
        self.policy = None


    def extend_node(self, node : Node):
        next_state = node.next_state()
        self.extend_count += 1

        # Prédiction du NN
        input = torch.from_numpy(next_state).to(DEVICE)
        result = self.net(input).result[0]

        # Mask des coups invalides et softmax
        invalid_moves = gogame.invalid_moves(next_state)  # contain also the pass move
        result[invalid_moves == 1] = float('-inf')
        prior_probs = torch.softmax(result, dim=-1)

        for action, p in enumerate(prior_probs):
            if p != 0 : 
                node.next_nodes.append(Node(
                    next_state,
                    action,
                    p,
                    parent = node,
                    depth = node.depth + 1
                ))


    def back_prop(self, node: Node, value, roll):
        node.back_propagation += 1

        node.N += -var.VIRTUAL_LOSS + 1
        node.Wr += 3 + roll
        node.Wv += value
        node.update_Q()

        if not node.is_root: 
            self.back_prop(node.parent, -value, -roll)

        return


    def back_prop_leaf(self, node: Node, value, roll, count : int):
        self.roll_forward_count += count
        node.roll = roll
        node.value = value
        self.back_prop(node, value, roll)


    def push_search(self, node: Node):
        node.front_push += 1

        node.N += var.VIRTUAL_LOSS
        node.Wr -= 3

        # Cas de la première exploration
        if node.first_search:
            node.first_search = False
            next_state = node.next_state()
            try:
                self.task_queue.put((node, next_state), timeout=3)    # node.value = self.net(next_state).value  and   node.roll = roll_policy(next_state)
                self.roll_policy_count += 1
            except queue.Full:
                assert False, "takes too much time, may be because the child thread is dead"

        # Cas N >= threshold
        elif node.N >= var.N_THRESHOLD and not node.end_game:
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
            self.back_prop(node, node.value, node.roll)

        return


    def find_node(self, action):
        """find the corresponding node of the action"""
        assert self.root.next_nodes, "the node need to be extended"
        for node in self.root.next_nodes :
            if node.action == action:
                return node
        assert False, "searching for a node which is an invalide move"


    def best_next_node(self):
        """Retourne l'enfant avec le plus grand nombre de visites N"""
        assert self.root.next_nodes, "Error : No policy is possible without expending this node"

        if self.root.Wr/self.root.N < -0.99 :
            print("resigne")
            return None

        return max(self.root.next_nodes, key=lambda n: n.N)
    

    def next_node(self):
        """Échantillonne un enfant selon la policy"""
        if self.policy is None:
            self.tree_policy()

        assert var.BOARD_SIZE**2 + 1 == np.count_nonzero(np.array(self.count_front) == np.array(self.count_back)), "a mismatch for the front push and for the backprop"
            
        ## print info
        # print("- Node selection info -")
        # print("Q of root : ", self.root.Q)
        # print("Wr of root : ", self.root.Wr)
        # print("Wv of root : ", self.root.Wv)
        # print("N of root : ", self.root.N)
        # self.affiche_policy()
        ##

        if self.root.Wr/self.root.N < -0.95 :
            print("resigne")
            return None

        r = random.random()
        cumulative = 0
        for node in self.root.next_nodes:
            cumulative += self.policy[node.action]
            if cumulative > r:
                # print("action taken", node.action_2d())
                self.policy = None
                return node

        assert False, "calculation error, self.policy don't sum to 1 or r = random.random() is not in the range (0,1)"
    

    def tree_policy(self):
        visited_N = [0 for i in range(var.BOARD_SIZE**2 + 1)]

        #-- mainly for debug plot --#
        count_front = [0 for i in range(var.BOARD_SIZE**2 + 1)]
        count_back = [0 for i in range(var.BOARD_SIZE**2 + 1)]
        for node in self.root.next_nodes:
            visited_N[node.action] = node.N
            count_front[node.action] = node.front_push
            count_back[node.action] = node.back_propagation

        self.count_branch_visit = visited_N  
        self.count_front = count_front
        self.count_back = count_back
        #----#

        visited_N_temperated = [N**(1 / self.temperature) for N in visited_N]
        diviseur = sum(visited_N_temperated)
        self.policy = [N / diviseur for N in visited_N_temperated]
        return self.policy


    def affiche_policy(self):
        tensor = torch.tensor(self.count_branch_visit[:-1])      # tensor 1D
        tensor_2d = tensor.view(var.BOARD_SIZE, var.BOARD_SIZE)

        print("- tree policy info -", flush=True)
        print("policy : ", tensor_2d, flush=True)
        print("pass : ", self.count_branch_visit[-1], flush=True)


    def __str__(self):
        return self.root.__str__()

