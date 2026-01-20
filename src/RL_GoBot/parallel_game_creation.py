import torch
import time
from math import sqrt
import random
from typing import List
from multiprocessing import Pool, Semaphore, pool

from RL_GoBot.rollout import roll_policy, init_roll_policy
from RL_GoBot.model import GoBot
from RL_GoBot import var
from RL_GoBot.data_base import GoDatabase

from gym_go import gogame, govars





def action2d(action) :
    return action // var.BOARD_SIZE, action % var.BOARD_SIZE 

    

class Node:
    def __init__(self, state, action: int, parent: "Node", p: torch.Tensor, depth=0, root=False):
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
                    node,
                    p,
                    node.depth + 1
                ))


    def back_prob(self, node: Node, value, roll):
        node.N += -var.VIRTUAL_LOSS + 1
        node.Wr += 3 + roll
        node.Wv += value
        # node.Q = ((1 - var.QU_RATIO) * node.Wv + var.QU_RATIO * node.Wr) /node.N
        node.update_Q()

        if node.root:       # we dont make the root Node parametters evolve 
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



def init_push_search(model):
    global net
    net = model


def one_self_play_MCTS(net : GoBot):
    with torch.no_grad():
        data_set = []
        state = gogame.init_state(var.BOARD_SIZE)
        root_node = Node(state, None, None, 1, root = True)
        tree = MCTS(net, root_node)
        while not gogame.game_ended(state) : 
            
            print("\n - new root - \n", tree)
            tmp = time.time()
            
            tree.sem = Semaphore(var.MAX_PROCESSES)
            with Pool(processes=var.MAX_PROCESSES, initializer=init_roll_policy, initargs=(net,)) as pool:
                for i in range(var.N_TREE_SEARCH) : 
                    tree.sem.acquire()
                    tree.push_search(tree.root, pool)

                pool.close()
                pool.join()
            
            MCTS_policy = tree.tree_policy()
            data_set.append([state, MCTS_policy])
        
            print("- general tree time and process info -", flush=True)
            print("nomber of rollout : ", MCTS.roll_policy_count, flush=True)
            print("nomber of forward : ", GoBot.forward_count, flush=True)
            print("time for this move : ", time.time() - tmp, flush=True)

            next_root = tree.next_node()
            next_root.root = True
            next_state = gogame.next_state(state, next_root.action)
            tree.root = next_root
            state = next_state
            MCTS.roll_policy_count = 0
            GoBot.forward_count = 0

    reward = gogame.winning(state, var.KOMI)
    for move in data_set:
        move.append(reward)
        reward = -reward

    return data_set


def self_play_MCTS(N, net : GoBot, db : GoDatabase):    # obliged when playing more then one game to save them in a db
    net.eval()
    for i in range(N):
        tmp_total = time.time()
        game_moves = one_self_play_MCTS(net)
        db.save_one_game(game_moves)
        
        print("     game -{}-".format(i))
        print("number of moves {}".format(len(game_moves)))
        print("duration of total game creation : {}".format(time.time() - tmp_total))
    return



if __name__ == "__main__":

    net = GoBot()
    one_self_play_MCTS(net)