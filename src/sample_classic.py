
from RL_GoBot.model import GoBot
from RL_GoBot.data_base import GoDatabaseMongo
from RL_GoBot.classic_game_creation import self_play_MCTS
from config import MODEL_DIR_9X9, GAMES_DIR

TYPE = "classic"
ID = 0
GENERATION = f"generation_{ID}" 
EPISODE = f"episode_{ID}"

# model load
net = GoBot()
net.load_model("{}/{}.pth".format(MODEL_DIR_9X9/TYPE, GENERATION))

# data base conection
db = GoDatabaseMongo(db_name= TYPE, collection_name = EPISODE)    # collection_name will be different for each of the episodes of the bot learning

# sampling
self_play_MCTS(100, net, db)