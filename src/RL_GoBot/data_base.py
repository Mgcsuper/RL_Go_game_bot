from pymongo import MongoClient
import lmdb
import io
from pathlib import Path

from torch.utils.data import Dataset, DataLoader
import torch


class GoDatabaseMongo(Dataset):
    def __init__(self, uri="mongodb://localhost:27017", db_name="go_rl", collection_name = "test"):
        self.collection_name = collection_name
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.counters = self.db["counters"]

        if self.counters.find_one({"_id": self.collection_name}) is None:
            self.counters.insert_one({"_id": self.collection_name, "seq": 0})

        self._len = len(self)

    def get_next_sequence(self):
        id = len(self)
        self.counters.update_one(
            {"_id": self.collection_name},
            {"$inc": {"seq": 1}},
        )
        return id

    def save_one_move(self, state : torch.tensor, policy : torch.tensor, reward : torch.tensor):
        doc = {
            "_id": self.get_next_sequence(),
            "state": state.tolist(),
            "policy": policy.tolist(),
            "reward": reward.item(),
        }
        self.collection.insert_one(doc)


    def save_one_game(self, game):   # game should be a list of moves, with a move being a list like [state, policy, reward]
        for move in game:
            self.save_one_move(move[0], move[1], move[2])

    
    def __len__(self):
        return self.counters.find_one({"_id": self.collection_name})["seq"]
    
    def __getitem__(self, idx):
        move = self.collection.find_one({"_id": idx})   # ATTENTION pas de +1 d'habitude
        state = torch.tensor(move["state"], dtype=torch.float32)
        policy = torch.tensor(move["policy"], dtype=torch.float32)
        reward = torch.tensor(move["reward"], dtype=torch.float32)
        return state, policy, reward
    
    def __iter__(self):
        for doc in self.collection.find({}):
            state = torch.tensor(doc["state"], dtype=torch.float32)
            policy = torch.tensor(doc["policy"], dtype=torch.float32)
            reward = torch.tensor(doc["reward"], dtype=torch.float32)
            yield state, policy, reward
    



class GoDatabaseLMDB(Dataset):
    def __init__(self, path: str| Path, db_name: str, map_size: int = 1_000_000_000):
        """
        path : Path to the LMDB file
        map_size : max size of the LMDB (in bit)
        """
        self.path = path
        self.map_size = map_size
        self.db_name = db_name

        self.env : lmdb = None 
        self.db = None
        self.count_db = None
        self.length = None


    def _update_length(self):   # need to be call by save_one_move()
        self.length += 1
        with self.env.begin(write=True, db=self.count_db) as txn:
            txn.put(self.db_name.encode("utf-8"), self.length.to_bytes(8, "big"))


    def _open(self):
        if self.env is None:
            self.env = lmdb.open(str(self.path), map_size=self.map_size, max_dbs=50)
            self.db = self.env.open_db(self.db_name.encode("utf-8"))
            self.count_db = self.env.open_db(b'counter')
            len(self)


    def _save_one_move(self, move):  # need to be call by save_one_game()
        """
        save one move (state, policy, reward) in LMDB
        """
        key = len(self).to_bytes(8, "big")  

        buffer = io.BytesIO()
        torch.save(move, buffer)
        data = buffer.getvalue()

        with self.env.begin(write=True, db=self.db) as txn:
            txn.put(key, data)
        
        self._update_length()


    def save_one_game(self, moves: list):
        """
        moves : liste de tuples (state, policy, reward)
        """
        self._open()
        for move in moves:
            self._save_one_move(move)

    def __len__(self):
        if self.length is None :
            self._open()
            with self.env.begin(db=self.count_db) as txn:
                length_bytes = txn.get(self.db_name.encode("utf-8"))
                if length_bytes:
                    self.length = int.from_bytes(length_bytes, "big")
                else:
                    self.length = 0

        return self.length
    

    def __getitem__(self, idx: int):
        key = idx.to_bytes(8, "big")
        self._open()
        with self.env.begin(db=self.db) as txn:
            data = txn.get(key)
            if data is None:
                raise IndexError(f"Index {idx} out of range")

            tensor_byte = io.BytesIO(data)
            move = torch.load(tensor_byte)

            return move
        

    def __iter__(self):
        self._open()
        with self.env.begin(db=self.db) as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                tensor_byte = io.BytesIO(value)
                move = torch.load(tensor_byte)
                yield move


    def drop(self):
        self._open()
        with self.env.begin(db=self.db, write=True) as txn:
            txn.drop(self.db, delete=True) 

        with self.env.begin(db=self.count_db, write=True) as txn:
            txn.delete(self.db_name.encode("utf-8"))

        with self.env.begin() as txn:
            cursor = txn.cursor()
            print("Clés dans la default DB (peuvent correspondre aux DBs nommées) :")
            for key, value in cursor:
                print(key) 



def get_data(db : Dataset, batch_size):
    return DataLoader(db, batch_size, shuffle=True) 