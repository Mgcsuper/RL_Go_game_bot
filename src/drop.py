from RL_GoBot.data_base import GoDatabaseLMDB
from config import GAMES_DIR

ORIGINE_DB = "batch"
EPISODE = 3
DESTINATION_GENERATION = f"generation_{EPISODE}" 

lmdb = GoDatabaseLMDB(path=GAMES_DIR/ORIGINE_DB, db_name=DESTINATION_GENERATION)

lmdb.drop()