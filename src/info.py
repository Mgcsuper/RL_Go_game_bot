from RL_GoBot.data_base import GoDatabaseLMDB, GoDatabaseMongo
from config import GAMES_DIR

ORIGINE_DB = "batch"
EPISODE = 0

DESTINATION_GENERATION = f"info_{EPISODE}" 

lmdb = GoDatabaseLMDB(path=GAMES_DIR/ORIGINE_DB, db_name=DESTINATION_GENERATION)
lmdb._open()

with lmdb.env.begin() as txn:
    cursor = txn.cursor()
    print("Clés dans la default DB (peuvent correspondre aux DBs nommées) :")
    for key, value in cursor:
        print(key) 

with lmdb.env.begin(db=lmdb.count_db) as txn:
    cursor = txn.cursor()
    print("conteur :")
    for key, value in cursor:
        print(key, int.from_bytes(value, "big")) 