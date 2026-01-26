from RL_GoBot.data_base import GoDatabaseLMDB, GoDatabaseMongo
from config import GAMES_DIR
import lmdb

ORIGINE_DB = "batch"
EPISODE = 0
MAP_SIZE = 100_000_000

env = lmdb.open(str(GAMES_DIR/ORIGINE_DB), map_size=MAP_SIZE, max_dbs=50)
count_db = env.open_db(b'counter')

with env.begin() as txn:
    cursor = txn.cursor()
    print("Clés dans la default DB (peuvent correspondre aux DBs nommées) :")
    for key, value in cursor:
        print(key) 

with env.begin(db=count_db) as txn:
    cursor = txn.cursor()
    print("conteur :")
    for key, value in cursor:
        print(key, int.from_bytes(value, "big")) 