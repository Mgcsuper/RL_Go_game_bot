from pymongo import MongoClient
import numpy as np


class GoDatabase:
    def __init__(self, uri="mongodb://localhost:27017", db_name="go_rl"):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.episodes = self.db["episodes"]

    def save_one_move(self, episode : int, state : np.array, policy : list, reward : int):
        doc = {
            "episode": episode,
            "state": state.tolist(),
            "policy": policy,
            "reward": reward
        }
        self.episodes.insert_one(doc)

    def save_one_game(self, episode : int, game):   # game should be a list of moves, with a move being a list [state, policy, reward]
        for move in game:
            self.save_one_move(episode, move[0], move[1], move[2])

    def load_episodes(self, limit=1000):
        return list(self.episodes.find().limit(limit))
