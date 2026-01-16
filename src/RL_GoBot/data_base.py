from pymongo import MongoClient
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch


class GoDatabase(Dataset):
    def __init__(self, uri="mongodb://localhost:27017", db_name="go_rl", collection_name = "test"):
        self.collection_name = collection_name
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        # self.collection = MongoClient(uri)[db_name][collection_name]  # équivalent
        self.counters = self.db["counters"]

        if "counters" not in self.db.list_collection_names():
            self.counters.insert_one({"_id": self.collection_name, "seq": -1})
        else:
            if self.counters.find_one({"_id": self.collection_name}) is None:
                self.counters.insert_one({"_id": self.collection_name, "seq": -1})

    def get_next_sequence(self):
        ret = self.counters.find_one_and_update(
            {"_id": self.collection_name},
            {"$inc": {"seq": 1}},
            return_document=True
        )
        return ret["seq"]

    def save_one_move(self, state : np.array, policy : list, reward : int):
        doc = {
            "_id": self.get_next_sequence(),
            "state": state.tolist(),
            "policy": policy,
            "reward": reward
        }
        self.collection.insert_one(doc)

    def save_one_game(self, game):   # game should be a list of moves, with a move being a list like [state, policy, reward]
        for move in game:
            self.save_one_move(move[0], move[1], move[2])

    # def load_collection(self, limit=1000):
    #     return list(self.collection.find().limit(limit))
    
    def __len__(self):
        return self.counters.find_one({"_id": self.collection_name})["seq"] + 1
    
    def __getitem__(self, idx):
        move = self.collection.find_one({"_id": idx})   # ATTENTION pas de +1 d'habitude
        state = torch.tensor(move["state"])
        policy = torch.tensor(move["policy"])
        reward = torch.tensor(move["reward"])
        return state, policy, reward



def get_data(db : GoDatabase, batch_size):
    return DataLoader(db, batch_size, shuffle=True) 