import numpy as np
import torch 
import time 
import queue

from RL_GoBot.batch_MCTSearch import Node, MCTS
from RL_GoBot.model import GoBot
from RL_GoBot import var
from config import DEVICE


from gym_go import gogame
from gym_go import govars



class Continuos_Rollout():
    def __init__(self, tree : MCTS):
        self.tree = tree        # for tree.net, tree.task_queue and tree.back_prop_leaf(node, value, roll, count)
        # self.batch_states = np.zeros((var.ROLL_BATCH_SIZE, govars.NUM_CHNLS, var.BOARD_SIZE, var.BOARD_SIZE), dtype=np.float32)
        # self.batchs_active = np.zeros((self.batch_size), dtype=bool)
        self.batch_size = var.ROLL_BATCH_SIZE

        self.gbatch_states = torch.zeros((var.ROLL_BATCH_SIZE, govars.NUM_CHNLS, var.BOARD_SIZE, var.BOARD_SIZE), dtype=torch.float32, device=DEVICE)
        self.gbatchs_active = torch.zeros((self.batch_size), dtype=torch.bool, device=DEVICE)
        self.gbatch_game_move_count = torch.zeros((self.batch_size), device=DEVICE)
        
        self.batch_states = np.zeros((var.ROLL_BATCH_SIZE, govars.NUM_CHNLS, var.BOARD_SIZE, var.BOARD_SIZE), dtype=np.float32)
        self.batchs_active = np.zeros((self.batch_size), dtype=bool)
                                          
        self.first_roll = np.zeros((self.batch_size), dtype=bool)
        self.batch_begining_values = np.zeros((self.batch_size))
        self.batch_origine_node = np.empty(self.batch_size, dtype=Node)
        self.batch_initial_turn = np.zeros((self.batch_size))   #gogame.batch_initial_turn(batch_states)
        self.batch_forward_count = np.zeros((self.batch_size), dtype=int)
        self.batch_game_ended = np.zeros((self.batch_size))
        self.end_tree_search = False
        self.active_count = 0


    def load_task(self):
        tmp = time.time()
        while self.active_count < var.ROLL_BATCH_SIZE and time.time() - tmp < var.MAX_WAIT_TIME and not self.end_tree_search : 
            origine_node : Node
            state : np.ndarray
            try :
                task  = self.tree.task_queue.get(timeout=0.001)      # task = (origini_node : Node, state : np.ndarray)
            except queue.Empty:
                task = False
            
            if task is None:
                self.end_tree_search = True

            elif task:
                self.active_count += 1
                (origine_node, state) = task
                idx = int(np.argmax(self.batchs_active == 0))
                self.batch_origine_node[idx] = origine_node
                self.batch_states[idx] = state
                self.batch_initial_turn[idx] = gogame.turn(state)
                self.batch_forward_count[idx] = 0
                self.batchs_active[idx] = True
                self.first_roll[idx] = True

                self.gbatchs_active[idx] = True
                self.gbatch_game_move_count[idx] = origine_node.depth
        return


    def batch_one_rollout(self):
        # print("batchs_active : ", self.batchs_active)
        # print("move_count", self.batch_game_move_count)
        gbatch_actions = - torch.ones((self.batch_size), dtype=torch.long, device=DEVICE) # * var.BOARD_SIZE**2
        self.gbatch_states = torch.from_numpy(self.batch_states).to(DEVICE)

        # net batch forward
        gbatch_net_output = self.tree.net(self.gbatch_states)
        batch_values = gbatch_net_output[:,-1].detach().cpu()
        gbatch_net_policy = gbatch_net_output[:,:-1]

        # compute invalid moves on GPU
        gbatch_invalid_moves = torch.zeros((self.batch_size, var.BOARD_SIZE**2 + 1), dtype=torch.bool, device=DEVICE)
        gbatch_invalid_moves[:, :-1] = self.gbatch_states[:, govars.INVD_CHNL].reshape(self.batch_size, -1)

        # register the value output of the net if it is the first move simulated by the rollout | or can use self.batch_forward_count == 0 as mask
        self.batch_begining_values[self.first_roll] = batch_values[self.first_roll]
        self.first_roll[:] = False

        # print("batch_net_policy : ", batch_net_policy)

        # creat batch_actions
        gbatch_net_policy[gbatch_invalid_moves] = -float("inf")
        gbatch_net_policy[~self.gbatchs_active] = -float("inf")
        gbatch_actions[self.gbatchs_active] = torch.argmax(gbatch_net_policy[self.gbatchs_active], dtype = torch.float32, dim=1)    

        # print("batch_net_policy : ", batch_net_policy)
        # print("batch_actions : ", batch_actions)    

        # case the game has more then var.MAX_TURNS moves
        gbatch_max_move = self.gbatch_game_move_count > (var.MAX_TURNS + var.ROLL_OFFSET)
        gbatch_actions[gbatch_max_move] = var.BOARD_SIZE**2
        # print("batch_actions : ", batch_actions)  

        # next_state
        batch_actions = gbatch_actions[self.gbatchs_active].detach().cpu().numpy()
        self.batch_states[self.batchs_active] = gogame.batch_next_states(self.batch_states[self.batchs_active], batch_actions)
        self.gbatch_game_move_count[self.gbatchs_active] += 1
        self.batch_forward_count[self.batchs_active] += 1


    def free(self):
        for i in range(self.batch_size):
            if self.batch_game_ended[i] and self.batchs_active[i]:
                score = gogame.winning(self.batch_states[i], var.KOMI)  # from black perspective
                # the turn of the first state of the rollout, correspond to the next_state(state, action) of the corresponding node.
                # Sinci it's the player of this next_state that should take the reward, if it is black the reward don't change
                if self.batch_initial_turn[i] :     
                    score = - score
                self.batchs_active[i] = False
                self.active_count -= 1
                self.tree.back_prop_leaf(self.batch_origine_node[i], self.batch_begining_values[i], score, self.batch_forward_count[i])


    def continus_batch_roll_out(self):
        with torch.no_grad():
            self.tree.net.eval()
            self.tree.net.to(device=DEVICE)
            while (not self.end_tree_search) or (self.active_count != 0):
                # batch creation
                if not self.end_tree_search :
                    # print("load")
                    self.load_task()
                        
                # roll_out
                self.batch_game_ended = gogame.batch_game_ended(self.batch_states)
                if np.all(self.batch_game_ended[self.batchs_active] == 0):
                    self.batch_one_rollout()
                    
                # if one of the rollout is ended, free the place 
                else :
                    self.free()
            
            assert self.active_count == 0, "the rollout wasn't finished"
            assert self.end_tree_search, "the parent thread hasn't send the ending event in the queue"

            self.end_tree_search = False
            return
        
