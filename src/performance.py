import time
import torch

from gym_go.envs import go_env
from gym_go import gogame

from RL_GoBot import var
from RL_GoBot.model import GoBot
from RL_GoBot.classic_MCTSearch import MCTS, Node, action1d


MODEL_PATH_1 = "batch/generation_2.pth"
MODEL_PATH_2 = "batch/generation_3.pth"
REFLEXION_TIME = 3  # in second

player_1 = GoBot()
player_1.load_model(MODEL_PATH_1)
player_2 = GoBot()
player_2.load_model(MODEL_PATH_2)

env = go_env.GoEnv(size=var.BOARD_SIZE, komi=var.KOMI)
init_node = Node(env.state(), None, 1)

tree_1 = MCTS(player_1, init_node)
tree_2 = MCTS(player_2, init_node)

def chose_action(tree, reflexion_time = REFLEXION_TIME) -> Node :
    with torch.no_grad():
        tmp = time.time()
        i = 0
        while (time.time() - tmp < reflexion_time) or (i <= 100) :
            i+=1
            tree.push_search(tree.root)

        print(time.time() - tmp)
        print(i)
        print("len : ", len(tree.root.next_nodes))

        next_node = tree.best_next_node()
        return  next_node

chose_action(tree_2, 1)


# Game loop
done = False
while True:
    # node selection
    chosen_node = chose_action(tree_1, 3)
    print("player_1 :\n", chosen_node)

    # env update
    state, reward, done, info = env.step(chosen_node.action)
    env.render()

    # tree update
    tree_1.root = chosen_node
    next_node_2 = tree_2.find_node(chosen_node.action)
    tree_2.root = next_node_2
    
    if env.game_ended():
        break

    # node selection
    chosen_node = chose_action(tree_2, 3)
    print("player_2 :\n", chosen_node)

    # env update
    state, reward, done, info = env.step(chosen_node.action)
    env.render()

    # tree update
    tree_2.root = chosen_node
    next_node_1 = tree_1.find_node(chosen_node.action)
    tree_1.root = next_node_1

    if env.game_ended():
        break


black_area, white_area = gogame.areas(state)
area_difference = black_area - white_area
komi_correction = area_difference - var.KOMI

if env.winning() == 1:
    print("{} win by {} points".format(MODEL_PATH_1, komi_correction))
    print("{} lose".format(MODEL_PATH_2))
elif env.winning() == -1:
    print("{} win by {} points".format(MODEL_PATH_2, -komi_correction))
    print("{} lose".format(MODEL_PATH_1))