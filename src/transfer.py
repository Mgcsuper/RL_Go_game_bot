from RL_GoBot.data_base import GoDatabaseLMDB, GoDatabaseMongo
from config import GAMES_DIR

ORIGINE_DB = "batch"
EPISODE = 3
ORIGINE_COL = f"episode_{EPISODE}" 

DESTINATION_GENERATION = f"generation_{EPISODE}" 


mongo = GoDatabaseMongo(db_name=ORIGINE_DB, collection_name=ORIGINE_COL) 
lmdb = GoDatabaseLMDB(path=GAMES_DIR/ORIGINE_DB, db_name=DESTINATION_GENERATION)


for move in mongo:
    lmdb.save_one_move(move)

print(len(lmdb))
print(len(mongo))