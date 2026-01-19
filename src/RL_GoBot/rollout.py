import numpy as np
import torch 
import time 
import queue

from RL_GoBot.batch_MCTSearch import Node, MCTS
from RL_GoBot.model import GoBot
from RL_GoBot import var


from gym_go import gogame
from gym_go import govars


global_net : GoBot = None

def init_roll_policy(model):
    global global_net
    global_net = model

def roll_policy(state : np.ndarray):    
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
                result = global_net(state).result[0]
                # print(result)
                sorted_indices_actions = sorted(range(len(result)), key=lambda i: result[i])    # sort from the most willing move to play until the most unwilling move, by the actual policy
                invalid_moves = gogame.invalid_moves(state)
                for i in range(var.BOARD_SIZE**2+1) :
                    action = sorted_indices_actions[i]
                    if not invalid_moves[action] :  # check if it is a valide move
                        break
        else :
            action = var.BOARD_SIZE**2
        
        state = gogame.next_state(state, action)
        game_ended = gogame.game_ended(state)
        game_turns += 1

    count = GoBot.forward_count  
    GoBot.forward_count  = 0

    score = gogame.winning(state, var.KOMI)  # from absolut black perspective
    if player :
        score = - score
    
    return score, count       # from the perspective of the player who where supposed to play at the starting state of the roll_policy
 


############################



class Continuos_Rollout():
    def __init__(self, tree : MCTS):
        self.tree = tree        # for tree.net, tree.task_queue and tree.back_prob_leaf(node, value, roll, count)
        self.batch_states = np.zeros((var.ROLL_BATCH_SIZE, govars.NUM_CHNLS, var.BOARD_SIZE, var.BOARD_SIZE), dtype=np.float32)
        self.batch_size = var.ROLL_BATCH_SIZE
        self.batchs_active = np.zeros((self.batch_size), dtype=bool)
        self.batch_origine_node = np.empty(self.batch_size, dtype=Node)
        self.batch_initial_turn = np.zeros((self.batch_size))   #gogame.batch_initial_turn(batch_states)
        self.batch_game_move_count = np.zeros((self.batch_size))
        self.batch_begining_values = np.zeros((self.batch_size))
        self.batch_game_ended = np.zeros((self.batch_size))
        self.end_tree_search = False
        self.active_count = 0



    def load_task(self):
        # print("load")
        # print("actives avant", self.batchs_active)
        tmp = time.time()
        while self.active_count < var.ROLL_BATCH_SIZE and time.time() - tmp < var.MAX_WAIT_TIME and not self.end_tree_search : 
            try :
                task  = self.tree.task_queue.get(timeout=0.001)      # task = (origini_node : Node, state : np.ndarray)
                # print("get")
            except queue.Empty:
                task = False
            
            if task is None:
                print("end tree search")
                self.end_tree_search = True

            elif task:
                self.active_count += 1
                (origine_node, state) = task
                idx = np.argmax(self.batchs_active == 0)
                self.batchs_active[idx] = 1
                self.batch_origine_node[idx] = origine_node
                self.batch_states[idx] = state
                self.batch_initial_turn[idx] = gogame.turn(state)
                self.batch_game_move_count[idx] = 0
        
        # print("actives après", self.batchs_active)
        return


    def batch_one_rollout(self):
        batch_actions = np.ones((self.batch_size)) * var.BOARD_SIZE**2
        # net batch forward
        batch_net_output = self.tree.net(self.batch_states)
        batch_values = batch_net_output[:,-1]
        batch_net_policy = batch_net_output[:,:-1]
        batch_invalid_moves = gogame.batch_invalid_moves(self.batch_states)

        # print("states", self.batch_states[3])
        # print(batch_invalid_moves[3])

        # register the value output of the net if it is the first move simulated by the rollout
        mask = np.where(self.batch_game_move_count == 0, True, False)
        self.batch_begining_values[mask] = batch_values[mask]

        # creat batch_actions
        sorted_indices_action = np.argsort(batch_net_policy, axis=1)
        for b in range(self.batch_size):
            if self.batchs_active[b] :
                for i in range(var.BOARD_SIZE**2+1) :
                    action = sorted_indices_action[b,i]
                    if not batch_invalid_moves[b, action] :  # check if it is a valide move
                        batch_actions[b] = action
                        break
        
        # print("batch_action", batch_actions)

        # case the game has more then var.MAX_TURNS moves
        batch_max_move = np.where(self.batch_game_move_count > var.MAX_TURNS, True, False) 
        batch_actions[batch_max_move] = var.BOARD_SIZE**2

        # print("batch_action", batch_actions)

        self.batch_states[self.batchs_active] = gogame.batch_next_states(self.batch_states, batch_actions)[self.batchs_active]
        self.batch_game_move_count[self.batchs_active] += 1


    def free(self):
        for i in range(self.batch_size):
            if self.batch_game_ended[i] and self.batchs_active[i]:
                score = gogame.winning(self.batch_states[i], var.KOMI)
                if self.batch_initial_turn[i] :
                    score = - score
                self.batchs_active[i] = 0
                self.active_count -= 1
                self.tree.back_prob_leaf(self.batch_origine_node[i], self.batch_begining_values[i], score, self.batch_game_move_count[i])


    def continus_batch_roll_out(self):
        with torch.no_grad():
            self.tree.net.eval()
            i = 0
            while (not self.end_tree_search) or (self.active_count != 0):
                if not self.end_tree_search :
                    ## batch creation
                    self.load_task()
                        
                # roll_out
                self.batch_game_ended = gogame.batch_game_ended(self.batch_states)
                # print("states :", self.batch_states[1])
                # print("___")
                # print("batch_game_move_count : ", self.batch_game_move_count)
                # print("batch_game_ended : ", self.batch_game_ended)
                # print("batchs_active : ", self.batchs_active)
                # print("end_game ", self.batch_game_ended[self.batchs_active])
                # print(" ")
                if np.all(self.batch_game_ended[self.batchs_active] == 0):
                    # print("batch_one_rollout")
                    self.batch_one_rollout()
                    # print("end batch_one_rollout")
                    
                # if one of the rollout is ended, free the place 
                else :
                    # print("free")
                    self.free()
                # print("_")
            print("- batch rollout info -")
            print("end task : " , self.active_count)
            print(self.end_tree_search)
            self.end_tree_search = False
            return
        

if __name__ == "__main__" : 
    pass


    # import queue

    # with torch.no_grad():
    #     net = GoBot()
    #     tree = MCTS(net, None, None)
    #     tree.net.eval()
    #     tree.task_queue = queue.Queue(8)
    #     roll_worker = Continuos_Rollout(tree)

    #     init_state = gogame.init_state(var.BOARD_SIZE)
    #     action1 = 0
    #     action2 = 1
    #     action3 = 2
    #     Node1 = Node(init_state, action1, None, 1)
    #     Node2 = Node(init_state, action2, None, 1)
    #     Node3 = Node(init_state, action3, None, 1)
    #     next_state1 = gogame.next_state(init_state, action1)
    #     next_state2 = gogame.next_state(init_state, action2)
    #     next_state3 = gogame.next_state(init_state, action3)
    #     init_state[[0,2],:,:] = 1
    #     Nodef = Node(init_state, 81, None, 1)

    #     tree.task_queue.put((Node1, next_state1))
    #     tree.task_queue.put((Node2, next_state2))

    #     tmp = time.time()
    #     roll_worker.load_task()
    #     print(time.time() - tmp)
    #     print(roll_worker.batchs_active)

    #     for i in range(5):
    #         roll_worker.batch_one_rollout()
    #         roll_worker.batch_game_ended = gogame.batch_game_ended(roll_worker.batch_states)
        
    #     print(roll_worker.batch_states[3])
    #     print(roll_worker.batchs_active)
    #     print(roll_worker.batch_game_move_count)

    #     print("__second__")
    #     tree.task_queue.put((Node3, next_state3))

    #     tmp = time.time()
    #     roll_worker.load_task()
    #     print(time.time() - tmp)
    #     print(roll_worker.batchs_active)

    #     for i in range(5):
    #         roll_worker.batch_one_rollout()
    #         roll_worker.batch_game_ended = gogame.batch_game_ended(roll_worker.batch_states)

        
    #     print(roll_worker.batch_states[3])
    #     print(roll_worker.batchs_active)
    #     print(roll_worker.batch_game_move_count)
    #     print(roll_worker.batch_game_ended)

    #     print("__third__")
    #     tree.task_queue.put((Nodef, init_state))

    #     tmp = time.time()
    #     roll_worker.load_task()
    #     print(time.time() - tmp)
    #     print(roll_worker.batchs_active)

    #     for i in range(3):
    #         print(f"___{i}___")
    #         roll_worker.batch_one_rollout()
    #         roll_worker.batch_game_ended = gogame.batch_game_ended(roll_worker.batch_states)
    #         print(roll_worker.batch_states[3])
    #         print(roll_worker.batch_game_ended)

        
    #     print(roll_worker.batchs_active)
    #     print(roll_worker.batch_game_move_count)



    # roll_worker.batch_states = gogame.batch_init_state(var.ROLL_BATCH_SIZE, var.BOARD_SIZE)
    # roll_worker.batchs_active = np.ones((var.ROLL_BATCH_SIZE - 1), dtype = bool)
    # with torch.no_grad():
    #     tree.net.eval()
    #     for i in range(5):
    #         roll_worker.batch_one_rollout()
    #         print("\n", i, " on essay")
    #         print(roll_worker.batch_states[0])

