
from RL_GoBot import var
from RL_GoBot.model import GoBot
from RL_GoBot.learning import train_one_episode
from RL_GoBot.data_base import GoDatabaseLMDB
from config import GAMES_DIR, MODEL_DIR_9X9

import torch
import os

TYPE = "batch"


if os.path.exists("{}/launch_generation_{}.pth".format(MODEL_DIR_9X9/TYPE, 1)):
    print("existe")
else :
    print("existe pas")



