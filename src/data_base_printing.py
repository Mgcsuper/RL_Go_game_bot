from RL_GoBot.data_base import GoDatabaseLMDB, GoDatabaseMongo
from config import GAMES_DIR
from gym_go import gogame

import torch


ORIGINE_DB = "test"
db_name = f"launch_0" 


lmdb = GoDatabaseLMDB(path=GAMES_DIR/ORIGINE_DB, db_name=db_name)

i = 0
for move in lmdb:
    print(f"_{i}_")
    i+=1
    print(gogame.str(move[0].numpy()))
    print(move[1], move[2])
