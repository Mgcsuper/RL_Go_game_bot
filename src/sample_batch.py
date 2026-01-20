from RL_GoBot.model import GoBot
from RL_GoBot.data_base import GoDatabase
from RL_GoBot.batched_game_creation import self_play_MCTS

# model load
net = GoBot()
net.load_model("batch/episode_1.pth")

# data base conection
db = GoDatabase(db_name= "batch", collection_name = "episode_1")    # collection_name will be different for each of the episodes of the bot learning

# sampling
self_play_MCTS(100, net, db)