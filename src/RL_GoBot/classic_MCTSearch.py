import torch

from math import sqrt
import random

from gym_go import gogame

from RL_GoBot.Node import Node
from RL_GoBot.model import GoBot
from RL_GoBot import var

from RL_GoBot.classic_rollout import roll_policy




class MCTS:
    roll_policy_count = 0
    def __init__(self, net: GoBot, root_node: Node, temperature = var.TEMPERATURE_MCTS):
        self.root = root_node
        self.net = net
        self.temperature = temperature 
        self.root_depth = 0
        self.policy = None


    def extend_node(self, node):
        # print("\n -- extend --")
        # Simulation de la création des enfants
        node.Wv = node.N * node.value
        node.Wr = node.N * node.roll
        next_state = node.next_state()

        # Prédiction du NN
        result = self.net(next_state)[0].result

        # Mask des coups invalides et softmax
        invalid_moves = gogame.invalid_moves(next_state)  # contain also the pass move
        result[invalid_moves == 1] = float('-inf')
        # print(result[:-1].view(7,7))
        # print(result[-1])
        prior_probs = torch.softmax(result, dim=-1)
        # print(prior_probs[:-1].view(7,7))
        # print(prior_probs[-1])

        for action, p in enumerate(prior_probs):
            if p != 0 : 
                node.next_nodes.append(Node(
                    next_state,
                    action,
                    p,
                    depth=node.depth + 1
                ))
        # print(len(node.next_nodes))


    def push_search(self, node: Node, prt=False):
        # Cas de la première exploration
        if node.first_search:
            MCTS.roll_policy_count += 1
            node.first_search = False
            next_state = node.next_state()
            node.value = self.net(next_state)[0].value
            node.roll = roll_policy(next_state, self.net, prt)
            node.Q = (1 - var.QU_RATIO) * node.value + var.QU_RATIO * node.roll
        
        if node.end_game:
            node.N += 1
            return -node.value, -node.roll

        # Cas N >= threshold
        if node.N >= var.N_THRESHOLD:
            if not node.next_nodes:
                self.extend_node(node)

            # Sélection du next_node selon UCT
            node_to_search = max(
                node.next_nodes,
                key=lambda n: n.Q + var.C_PUCT * n.p * sqrt(node.N) / (1 + n.N)
            )
            # qpu = [n.Q + var.C_PUCT * n.p * sqrt(node.N) / (1 + n.N) for n in node.next_nodes]
            # print(qpu)
            # print(node_to_search)

            value, output = self.push_search(node_to_search, prt)
            node.Wv += value
            node.Wr += output
            node.update_Q()

        else:
            value = node.value
            output = node.roll

        node.N += 1
        return -value, -output


    def best_next_node(self):
        """Retourne l'enfant avec le plus grand nombre de visites N"""
        if not self.root.next_nodes:
            print("Error : No policy is possible without expending this node")
            return None
        best_node = max(self.root.next_nodes, key=lambda n: n.N)
        return best_node
    

    def find_node(self, action):
        """find the corresponding node of the action"""
        if not self.root.next_nodes :
            print("Erreur, search for a next node when there is no next nodes")
            self.extend_node(self.root)
        for node in self.root.next_nodes :
            if node.action == action:
                return node
        return None


    def next_node(self):
        """Échantillonne un enfant selon la policy"""
        if self.policy is None:
            self.tree_policy()
            
        r = random.random()
        self.affiche_policy()
        print("r : ", r)
        cumulative = 0
        for node in self.root.next_nodes:
            cumulative += self.policy[node.action]
            if cumulative > r:
                print("action : ", node.action_2d())
                self.policy = None
                return node

        self.policy = None
        # Au cas où la somme n'est pas exactement 1 à cause des flottants
        return self.next_nodes[-1]
    

    def tree_policy(self):
        visited_N = [0 for i in range(var.BOARD_SIZE**2 + 1)]
        for node in self.root.next_nodes:
            visited_N[node.action] = node.N

        self.count_branch_visit = visited_N  # juste pour de l'affichage

        visited_N_temperated = [N**(1 / self.temperature) for N in visited_N]
        diviseur = sum(visited_N_temperated)
        self.policy = [N / diviseur for N in visited_N_temperated]
        return self.policy


    def affiche_policy(self):
        tensor = torch.tensor(self.count_branch_visit[:-1])      # tensor 1D
        tensor_2d = tensor.view(var.BOARD_SIZE, var.BOARD_SIZE)  

        print("- tree policy info -")
        print("policy : ", tensor_2d)
        print("pass : ", self.count_branch_visit[-1])

    def __str__(self):
        return self.root.__str__()