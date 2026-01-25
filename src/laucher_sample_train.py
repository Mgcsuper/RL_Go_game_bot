from multiprocessing import Pool, Semaphore
import torch
import time

from RL_GoBot.model import GoBot
from RL_GoBot.data_base import GoDatabaseLMDB, get_data
from RL_GoBot.batch_game_creation import one_self_play_MCTS
from RL_GoBot.learning import get_optimizer, train
from RL_GoBot import var
from config import MODEL_DIR_9X9, GAMES_DIR


TYPE = "batch"
MAX_PROCESSES = 8
net = GoBot()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_episode(net : GoBot,
                            db : GoDatabaseLMDB,
                            batch_size=var.BATCH_SIZE,
                            device='cuda',
                            learning_rate=var.LEARNING_RATE,
                            weight_decay=var.L2_LOSS,
                            momentum=var.MOMENTUM,
                            epochs=var.EPOCHS):
            '''
            Input arguments
            batch_size: Size of a mini-batch
            device: GPU where you want to train your network
            weight_decay: Weight decay co-efficient for regularization of weights
            momentum: Momentum for SGD optimizer
            epochs: Number of epochs for training the network
            '''
        
            optimizer = get_optimizer(net, learning_rate, weight_decay, momentum)

            for e in range(epochs):
                print("epoch : ", e)
                train_loader = get_data(db, batch_size)
                train(net, train_loader, optimizer, device)



if __name__ == "__main__":

    for ID in range(10):
        GENERATION = f"launch_generation_{ID}" 
        EPISODE = f"launch_{ID}"

        print("generation {}".format(ID))
        print("db_sampling ...")

        # model load
        net.load_model("{}/{}.pth".format(MODEL_DIR_9X9/TYPE, GENERATION))

        # data base conection
        db = GoDatabaseLMDB(path= GAMES_DIR/TYPE, db_name= EPISODE)    # db_name will be different for each of the generation of the bot learning

        # semaphore
        sem = Semaphore(MAX_PROCESSES)

        def callback(game_moves):
            sem.release()
            print("save one game Nb moves : ", len(game_moves))
            db.save_one_game(game_moves)


        # sampling
        with Pool(processes=MAX_PROCESSES) as pool:
            db_size = 0 
            
            while db_size < 5000 :
                sem.acquire()
                pool.apply_async(one_self_play_MCTS, args=(net,), callback = callback) 
                db_size = len(db)

            pool.close()
            pool.join()
        

        # train
        print("training ...")

        train_one_episode(net, db, device = device)
        net.save_model("{}/launch_generation_{}.pth".format(MODEL_DIR_9X9/TYPE, ID + 1))