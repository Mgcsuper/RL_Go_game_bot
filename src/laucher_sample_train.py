from multiprocessing import Pool, Semaphore
import torch
import time
import os

from RL_GoBot.model import GoBot
from RL_GoBot.data_base import GoDatabaseLMDB
from RL_GoBot.batch_game_creation import one_self_play_MCTS
from RL_GoBot.learning import train_one_episode
from RL_GoBot import var
from config import MODEL_DIR_9X9, GAMES_DIR, DEVICE


TYPE = "batch"
MAX_PROCESSES = 8
INIT_ID = 0

TEMPERATURE_MCTS = var.TEMPERATURE_MCTS
DECREACE_TEMPERATURE_MCTS = var.DECREACE_TEMPERATURE_MCTS
LEARNING_RATE = var.LEARNING_RATE
DECREACE_LEARNING_RATE = var.DECREACE_LEARNING_RATE
MODEL_WEIGHT_TEMPERATURE = var.MODEL_WEIGHT_TEMPERATURE

net = GoBot()
 

if __name__ == "__main__":

    for ID in range(INIT_ID, 10):
        GENERATION = f"launch_generation_{ID}" 
        EPISODE = f"launch_{ID}"

        print("generation {}".format(ID))
        print("db_sampling ...")

        # model load
        net.load_model("{}/{}.pth".format(MODEL_DIR_9X9/TYPE, GENERATION))
        net.to(device=DEVICE)

        # data base conection
        db = GoDatabaseLMDB(path= GAMES_DIR/TYPE, db_name= EPISODE)    # db_name will be different for each of the generation of the bot learning

        # semaphore
        sem = Semaphore(MAX_PROCESSES)

        # callback
        def callback(game_moves):
            sem.release()
            print("save one game Nb moves : ", len(game_moves))
            db.save_one_game(game_moves)


        # sampling
        with Pool(processes=MAX_PROCESSES) as pool:
            db_size = len(db)
            
            while db_size < var.DATABASE_SIZE :
                sem.acquire()
                pool.apply_async(one_self_play_MCTS, args=(net, TEMPERATURE_MCTS * (DECREACE_TEMPERATURE_MCTS**ID)), callback = callback) 
                db_size = len(db)
            print("end sampling")
            pool.close()
            pool.join()
        

        # train
        print("training ...")

        if not os.path.exists("{}/launch_generation_{}.pth".format(MODEL_DIR_9X9/TYPE, ID + 1)):
            train_one_episode(net, 
                            db, 
                            learning_rate=LEARNING_RATE * (DECREACE_LEARNING_RATE**ID), 
                            temperature=MODEL_WEIGHT_TEMPERATURE*LEARNING_RATE * (DECREACE_LEARNING_RATE**(2*ID)))
            
            net.save_model("{}/launch_generation_{}.pth".format(MODEL_DIR_9X9/TYPE, ID + 1))
        else:
            print("model already train and saved")
