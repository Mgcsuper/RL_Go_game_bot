from RL_GoBot.model import GoBot
from RL_GoBot.data_base import GoDatabaseLMDB
from RL_GoBot.batch_game_creation import self_play_MCTS
from config import MODEL_DIR_9X9, GAMES_DIR

TYPE = "batch"
ID = 0
GENERATION = f"launch_generation_{ID}" 
EPISODE = f"launch_{ID}"

# model load
net = GoBot()
net.load_model("{}/{}.pth".format(MODEL_DIR_9X9/TYPE, GENERATION))

# data base conection
db = GoDatabaseLMDB(path= GAMES_DIR/TYPE, db_name= EPISODE)    # collection_name will be different for each of the episodes of the bot learning

# sampling
self_play_MCTS(100, net, db)