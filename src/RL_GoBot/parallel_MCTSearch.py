import torch
from math import sqrt
import random
from multiprocessing import pool

from RL_GoBot.parallel_rollout import roll_policy
from RL_GoBot.model import GoBot
from RL_GoBot.Node import Node
from RL_GoBot import var

from gym_go import gogame, govars



# def action2d(action) :
#     return action // var.BOARD_SIZE, action % var.BOARD_SIZE 



class MCTS:
    roll_policy_count = 0
    def __init__(self, net: GoBot, root_node : Node, temperature = var.TEMPERATURE_MCTS):
        self.sem = None
        self.root = root_node
        self.net = net
        self.temperature = temperature 
        self.root_depth = 0
        self.policy = None


    def extend_node(self, node):
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
                    p,
                    parent = node,
                    depth = node.depth + 1
                ))


    def back_prob(self, node: Node, value, roll):
        node.N += -var.VIRTUAL_LOSS + 1
        node.Wr += 3 + roll
        node.Wv += value
        # node.Q = ((1 - var.QU_RATIO) * node.Wv + var.QU_RATIO * node.Wr) /node.N
        node.update_Q()

        if node.is_root:       # we dont make the root Node parametters evolve 
            self.sem.release()
        else :
            self.back_prob(node.parent, -value, -roll)

        return


    def back_prob_leaf(self, node: Node, value, roll, count : int):
        GoBot.forward_count += count
        node.roll = roll
        self.back_prob(node, value, roll)


    def push_search(self, node: Node, pool : pool.Pool):
        node.N += var.VIRTUAL_LOSS
        node.Wr -= 3

        # Cas de la première exploration
        if node.first_search:
            MCTS.roll_policy_count += 1
            node.first_search = False
            next_state = node.next_state()
            node.value = self.net(next_state).value
            pool.apply_async(roll_policy, args=(next_state,), callback = lambda result : self.back_prob_leaf(node, node.value, result[0], result[1])) 
            

        # Cas N >= threshold
        elif node.N >= var.N_THRESHOLD and not node.end_game:
            if not node.next_nodes:
                self.extend_node(node)

            # Sélection du next_node 
            node_to_search = max(
                node.next_nodes,
                key=lambda n: n.Q + var.C_PUCT * n.p * sqrt(node.N) / (1 + n.N)
            )
            self.push_search(node_to_search, pool)

        # Cas 1 <= N < threshold
        else:
            self.back_prob(node, node.value, node.roll)
        
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

        print("- tree policy info -", flush=True)
        print("illegal : ", self.root.state[govars.INVD_CHNL])
        print("policy : ", tensor_2d, flush=True)
        print("pass : ", self.count_branch_visit[-1], flush=True)

    def __str__(self):
        return self.root.__str__()



if __name__ == "__main__":
    print("wooo MCTS")
    # import multiprocessing
    # multiprocessing.freeze_support()