import time

from gym_go.envs import go_env
from RL_GoBot import var
from RL_GoBot.model import GoBot
from RL_GoBot.classic_MCTSearch import MCTS, Node, action1d


MODEL_PATH = "batch/episode_2.pth"
REFLEXION_TIME = 3  # in second

bot = GoBot()
bot.load_model(MODEL_PATH)

env = go_env.GoEnv(size=var.BOARD_SIZE, komi=var.KOMI)

init_node = Node(env.state(), None, 1)
tree = MCTS(bot, init_node)

def chose_action(reflexion_time = REFLEXION_TIME) :
    tmp = time.time()
    i = 0
    print(time.time() - tmp)
    while time.time() - tmp < reflexion_time :
        i+=1
        print(time.time() - tmp)
        tree.push_search(tree.root)
        print(time.time() - tmp)
    
    print(i)
    print("len : ", len(tree.root.next_nodes))

    next_node = tree.best_next_node()
    tree.root = next_node
    return  next_node.action

chose_action(1)


# Game loop
done = False
while not done:
    valide_moves = env.valid_moves()
    action = env.render(mode = 'human')
    action_1d = action1d(action)
    while not valide_moves[action_1d] :
        print("invalid move")
        action = env.render(mode = 'human')
        action_1d = action1d(action)

    print(len(tree.root.next_nodes))
    state, reward, done, info = env.step(action)
    next_node = tree.find_node(action)
    tree.root = next_node
    

    if env.game_ended():
        break

    action = chose_action(3)
    print(action)
    state, reward, done, info = env.step(action)

    if env.game_ended():
        break
    
env.render()
