
# About
A Go game bot using a self-updated version of the **GymGo** environment based on the following repository:  
[GymGo](https://github.com/huangeddie/GymGo)

The overall strategy consists of:
1. Sampling games using the most recent generation of the AI model with MCTS  
2. Training the same model on the sampled games  

The long-term objective is to reach a strength of **50**.

For each generation, we reduce:
- the learning rate  
- the weight temperature used during training  
- the MCTS temperature  

There are three versions of the game-sampling algorithm. The **batch** version is the most efficient one, so it is used everywhere.

---

# Installation

Clone the repository and then:

```bash
# In the root directory
pip install uv
uv sync
```

# Run the code

```bash
# to run any file
uv run python file_name.py
```


# Folders Description

### Data
This folder is used for storing the bot model weights in the `model` folder.  
Game moves sampled in each generation are stored in the `game_moves` folder.  
The most efficient version of the code is the batch version, so the model and the game moves are stored in the `batch` subfolder.

### src
#### gym_go
The Go game environment imported from the GymGo repository.

#### RL_GoBot
Source code for the entire project:  
- `model.py` : defines the model architecture.  
- `Node.py` : tree nodes for the MCTS.  
- `MCTSearch.py` : the tree algorithm itself, including push search, rollout backpropagation, and tree expansion methods.  
- `game_creation.py` : manages the tree, the database, and the AI model used for sampling a game.  
- `data_base.py` : database classes.  
- `rollout.py` : executes the rollout of the policy (here, the AI model itself) from a leaf node of the MCTS to a terminal game state, including resignations.  

Files prefixed with `batch_` represent the most efficient way to sample a game.

#### Higher-Level Files
- To play against a specific generation bot, use `play.py`.  
- To create your own trained model, use `launcher_sample_train_PC.py` (training several generations can take several days).  
- To make different generations play against each other, use `play_performance_batch.py`.
