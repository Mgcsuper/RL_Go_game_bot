
from RL_GoBot import var
from RL_GoBot.model import GoBot
from RL_GoBot.learning import train_one_episode
from RL_GoBot.data_base import GoDatabaseLMDB
from config import GAMES_DIR, MODEL_DIR_9X9

import torch

TYPE = "batch"
ID = 0
GENERATION = f"test_{ID}" 
EPISODE = f"launch_{ID}"       # a better name would be generation



bot = GoBot()
bot.load_model("{}/{}.pth".format(MODEL_DIR_9X9/TYPE, GENERATION))

db = GoDatabaseLMDB(path= GAMES_DIR/TYPE, db_name= EPISODE)    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_one_episode(bot, db, device = device, temperature=(var.MODEL_WEIGHT_TEMPERATURE*var.LEARNING_RATE), epochs=3)

bot.save_model("{}/test_{}.pth".format(MODEL_DIR_9X9/TYPE, ID+1))


