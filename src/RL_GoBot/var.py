BOARD_SIZE = 7
KOMI = 0.5
MAX_TURNS = 100 # a bit plus then the total number of position to play

# MCTS variables
C_PUCT = 5
N_THRESHOLD = 10 
N_TREE_SEARCH = 100
QU_RATIO = 0.5

TEMPERATURE_MCTS = 2

# training variables
# DEVICE = 'cuda:0'
EPOCHS = 10
MOMENTUM = 0.9
LEARNING_RATE = 0.01
BATCH_SIZE = 16

# loss variables
L2_LOSS = 0.00001
CLASSIFICATION_LOSS = 1
REGRESSION_LOSS = 0.5
