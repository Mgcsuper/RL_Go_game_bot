from RL_GoBot.data_base import GoDatabaseLMDB
from config import GAMES_DIR


def test_db_lmdb():
    ORIGINE_DB = "batch"
    EPISODE = 3
    DESTINATION_GENERATION = f"episode_{EPISODE}" 

    lmdb = GoDatabaseLMDB(path=GAMES_DIR/ORIGINE_DB, db_name=DESTINATION_GENERATION)

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
