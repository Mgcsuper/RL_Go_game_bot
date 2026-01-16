import torch
from gym_go import gogame, govars
from RL_GoBot import var
from RL_GoBot.model import GoBot
from math import sqrt
import random



def action2d(action) :
    return action // var.BOARD_SIZE, action % var.BOARD_SIZE 


def roll_policy(state, net, prt = False):    
    """
    return :
        the score of the outcome of the self played game seen by the first player to play
    """
    player = state[2,0,0]
    game_ended = gogame.game_ended(state)

    game_turns = 0
    while not game_ended:
        if game_turns < var.MAX_TURNS :
            if torch.all(torch.tensor(state[3,:,:]) == 1):  # if there is no more legal mouves, the only move is pass
                action = var.BOARD_SIZE**2      # pass move
            else :
                result = net([state]).result[0]
                # print(result)
                sorted_indices_actions = sorted(range(len(result)), key=lambda i: result[i])    # sort from the most willing move to play until the most unwilling move, by the actual policy
                invalid_moves = gogame.invalid_moves(state)
                for i in range(var.BOARD_SIZE**2+1) :
                    action = sorted_indices_actions[i]
                    if not invalid_moves[action] :  # check if it is a valide move
                        break
        else :
            action = var.BOARD_SIZE**2
        
        # if prt : print(action2d(action))
        # if prt : print(state)
        state = gogame.next_state(state, action)
        game_ended = gogame.game_ended(state)
        game_turns += 1

    score = gogame.winning(state, var.KOMI)  # from absolut black perspective
    if player :
        score = - score
    return score        # from the perspective of the player who where supposed to play at the starting state of the roll_policy
 
    


class Node:
    def __init__(self, state, action: int, p: torch.Tensor, depth=0):
        self.state = state
        self.action = action
        self.p = p
        self.depth = depth
        self.end_game = self.next_state()[govars.DONE_CHNL,0,0]

        self.N = 0
        self.Wv = 0
        self.Wr = 0
        self.Q = 0

        self.next_nodes = []

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
        print("N : {}, P : {}".format(self.N, self.p))
        print(self.state[3])
        return ' '


class MCTS:
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
        result = self.net([next_state]).result[0]

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
                    node.depth + 1
                ))
        # print(len(node.next_nodes))


    def push_search(self, node: Node, prt=False):
        # Cas de la première exploration
        if node.first_search:
            node.first_search = False
            next_state = node.next_state()
            node.value = self.net(next_state).value
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
                print("action : ", node.action)
                self.policy = None
                return node

        self.policy = None
        # Au cas où la somme n'est pas exactement 1 à cause des flottants
        return self.next_nodes[-1]
    

    def tree_policy(self):
        visited_N = [0 for i in range(var.BOARD_SIZE**2 + 1)]
        for node in self.root.next_nodes:
            visited_N[node.action] = node.N

        visited_N_temperated = [N**(1 / self.temperature) for N in visited_N]
        diviseur = sum(visited_N_temperated)
        self.policy = [N / diviseur for N in visited_N_temperated]
        return self.policy


    def affiche_policy(self):
        tensor = torch.tensor(self.policy[:-1])      # tensor 1D
        tensor_7x7 = tensor.view(7, 7)  
        print("policy : ", tensor_7x7)
        print("pass : ", self.policy[-1])

    def __str__(self):
        print(self.root)
        return ' '