import queue

from gym_go.envs import go_env
from gym_go import gogame

from RL_GoBot import var
from RL_GoBot.model import GoBot
from RL_GoBot.Node import Node
from RL_GoBot.batch_MCTSearch import MCTS
from RL_GoBot.batch_game_creation import one_move_timed
from RL_GoBot.batch_rollout import Continuos_Rollout

MODEL_PATH_1 = "batch/launch_generation_2.pth"
MODEL_PATH_2 = "batch/launch_generation_10.pth"
REFLEXION_TIME = 2  # in second
MIN_TREE_PUSH = 100

player_1 = GoBot()
player_1.load_model(MODEL_PATH_1)
player_2 = GoBot()
player_2.load_model(MODEL_PATH_2)

env = go_env.GoEnv(size=var.BOARD_SIZE, komi=var.KOMI)
init_node_1 = Node(env.state(), None, 1, parent = None, root=True)
init_node_2 = Node(env.state(), None, 1, parent = None, root=True)

task_queue_1 = queue.Queue(maxsize=var.MAX_QUEUE_SIZE)
task_queue_2 = queue.Queue(maxsize=var.MAX_QUEUE_SIZE)

tree_1 = MCTS(player_1, init_node_1, task_queue_1)
tree_2 = MCTS(player_2, init_node_2, task_queue_2)
rollout_object_1 = Continuos_Rollout(tree_1)
rollout_object_2 = Continuos_Rollout(tree_2)



# Game loop
done = False
while True:
    # node selection
    one_move_timed(tree_1, rollout_object_1, REFLEXION_TIME, MIN_TREE_PUSH)
    one_move_timed(tree_2, rollout_object_2)
    chosen_node = tree_1.best_next_node()
    # resigne
    if chosen_node is None:
        break
    print("player_1 (black):\n", chosen_node)

    # env update
    state, reward, done, info = env.step(chosen_node.action)
    env.render()

    # tree update
    chosen_node.is_root = True
    tree_1.root = chosen_node
    next_node_2 = tree_2.find_node(chosen_node.action)
    next_node_2.is_root = True
    tree_2.root = next_node_2
    
    if env.game_ended():
        break

    # node selection
    one_move_timed(tree_2, rollout_object_2, REFLEXION_TIME, MIN_TREE_PUSH)
    one_move_timed(tree_1, rollout_object_1)
    chosen_node = tree_2.best_next_node()
    # resigne
    if chosen_node is None:
        break
    print("player_2 (white):\n", chosen_node)

    # env update
    state, reward, done, info = env.step(chosen_node.action)
    env.render()

    # tree update
    chosen_node.is_root = True
    tree_2.root = chosen_node
    next_node_1 = tree_1.find_node(chosen_node.action)
    next_node_1.is_root = True
    tree_1.root = next_node_1

    if env.game_ended():
        break


black_area, white_area = gogame.areas(state)
area_difference = black_area - white_area
komi_correction = area_difference - var.KOMI

if env.winning() == 1:
    print("black : {} win by {} points".format(MODEL_PATH_1, komi_correction))
    print("white : {} lose".format(MODEL_PATH_2))
elif env.winning() == -1:
    print("white : {} win by {} points".format(MODEL_PATH_2, -komi_correction))
    print("black : {} lose".format(MODEL_PATH_1))