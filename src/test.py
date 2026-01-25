from RL_GoBot import var
from RL_GoBot.model import GoBot
from RL_GoBot.learning import get_optimizer, compute_loss, train
from RL_GoBot.data_base import GoDatabaseLMDB, get_data

from config import GAMES_DIR

import torch

DATABASE = "batch"
EPISODE = 2
GENERATION = f"generation_{EPISODE}"       # a better name would be generation

bot = GoBot()
bot.load_model("{}/{}.pth".format(DATABASE, GENERATION))
lmdb = GoDatabaseLMDB(path=GAMES_DIR/DATABASE, db_name=GENERATION)

