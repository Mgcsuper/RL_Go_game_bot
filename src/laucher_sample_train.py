from RL_GoBot.model import GoBot
from RL_GoBot.data_base import GoDatabaseLMDB
from RL_GoBot.batch_game_creation import self_play_MCTS
from config import MODEL_DIR_9X9, GAMES_DIR

TYPE = "batch"


for EPISODE in range(10):
    GENERATION = f"generation_{EPISODE}" 

    # model load
    net = GoBot()
    net.load_model("{}/{}.pth".format(MODEL_DIR_9X9/TYPE, GENERATION))

    # data base conection
    db = GoDatabaseLMDB(path= GAMES_DIR/TYPE, db_name= GENERATION)    # db_name will be different for each of the generation of the bot learning

    db_size = 0
    # sampling
    while db_size < 5_000 :
        self_play_MCTS(100, net, db)