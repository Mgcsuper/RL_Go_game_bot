import torch
from gym_go import gogame
from RL_GoBot import var
from RL_GoBot.model import GoBot
from math import sqrt



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
    

class MCTS():
    root_depth = 0
    net : GoBot

    def __init__(self, state, action : int, p : torch.Tensor, depth = 0):
        self.next_nodes = []
        self.state = state
        self.action = action
        self.N = 0
        self.Wv = 0 
        self.Wr = 0 
        self.Q = 0
        self.p = p
        self.depth = depth

        self.first_search = True
        self.value = 0
        self.roll = 0

    def next_state(self):
        if self.action == None :
            return self.state
        else :
            return gogame.next_state(self.state, self.action)
    
    def action_2d(self):
        if self.action == None :
            return "None"
        else :
            return self.action // var.BOARD_SIZE, self.action % var.BOARD_SIZE


    def exctend_tree(self):
        # here the simulation of the accumulation of Wv and Wr will being just a leaf node
        self.Wv = self.N * self.value
        self.Wr = self.N * self.roll     
        next_state = self.next_state()
        result = MCTS.net([next_state]).result[0]
        for i in range(var.BOARD_SIZE):
            for j in range(var.BOARD_SIZE):
                if next_state[3,i,j] == 1:
                    p = 0
                else : 
                    p = max(var.EPSILON,result[i*var.BOARD_SIZE + j])
                action = i*var.BOARD_SIZE + j
                # adding the mouve at possition (i,j)
                self.next_nodes.append(
                    MCTS(next_state, 
                        action, 
                        p,
                        self.depth + 1))
        # adding the pass mouve
        self.next_nodes.append(
            MCTS(next_state, 
                var.BOARD_SIZE**2, 
                var.EPSILON,
                self.depth + 1))


    def push_search(self, prt = False):
        if prt : print("self", self)
        if self.N >= var.N_THRESHOLD :
            if not self.next_nodes :    # extend node part
                print("extende : ", self.action_2d(), self.depth)
                self.exctend_tree()

            max_score = 0      # all scores can be <0 so can be improved here (maybe take the score of the default pass mouve)
            # max_score = self.next_nodes[-1].Q + c_puct * self.next_nodes[-1].p * sqrt(self.N)/(1+self.next_nodes[-1].N)
            node_to_search = self.next_nodes[-1]    # default next mouve is the pass mouve (used if no mouves has a positive score)
            if prt : print("___scores___")
            for next_node in self.next_nodes :
                score = next_node.Q + var.C_PUCT * next_node.p * sqrt(self.N)/(1+next_node.N)
                if prt : print(score)
                # print(score, next_node)
                if score > max_score :
                    max_score = score
                    node_to_search = next_node
            if prt : print("___scores_end___")
            value, output = node_to_search.push_search(prt)
            self.Wv += value  
            self.Wr += output 
            self.Q = ((1-var.QU_RATIO) * self.Wv + var.QU_RATIO * self.Wr)/self.N
            

        elif self.first_search :    # here self.Q is initialise and dont change the N_threshold first time but self.Wv and self.Wr should accumulate with is simulated in the extend node part
            if prt : print("___first_search___")
            self.first_search = False
            next_state = self.next_state()
            value = MCTS.net(next_state).value
            if prt : print("__in_roll_out__")
            output = roll_policy(next_state, MCTS.net, prt)
            if prt : print("___first_search_end___")
            
            self.value = value
            self.roll = output
            self.Q = (1-var.QU_RATIO) * self.value + var.QU_RATIO * self.roll     # don't change value until the expend happen

        else :
            value = self.value
            output = self.roll
    
        self.N += 1
        return - value, - output
    

    def best_next_node(self):
        max_visited_N = 0
        i = 0
        for next_node in self.next_nodes :
            i += 1
            visited_N = next_node.N
            if visited_N > max_visited_N :
                max_visited_N = visited_N
                best_node = next_node
        return best_node
    

    @staticmethod
    def tree_policy(node):
        actions = []
        visited_N = []
        for next_node in node.next_nodes :
            actions.append(next_node.action)
            visited_N.append(next_node.N)
        diviseur = sum(N**(1/var.TEMPERATURE_MCTS) for N in visited_N)
        sorted_index = sorted(range(len(actions)), key=lambda i : actions[i])
        MCTS_policy = [visited_N[i]/diviseur for i in sorted_index]
        return MCTS_policy
    
    def __str__(self):
        print("___ node info ___")
        print("action : {} | depth : {}".format(self.action_2d(), self.depth))
        print(self.N)
        # print(self.state[3])
        return ' '
        

