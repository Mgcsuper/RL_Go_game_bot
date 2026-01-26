from RL_GoBot.data_base import GoDatabaseLMDB
from config import GAMES_DIR

TYPE = "batch"
db_name = "launch_1"

lmdb = GoDatabaseLMDB(path=GAMES_DIR/TYPE, db_name=db_name)

lmdb.drop()