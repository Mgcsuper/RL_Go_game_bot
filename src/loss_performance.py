import time
import torch

from RL_GoBot.model import GoBot
from RL_GoBot.learning import test
from RL_GoBot.data_base import GoDatabaseLMDB, get_data

from config import MODEL_DIR_9X9, GAMES_DIR

TYPE = "test_gen_2"
# MODEL_PATH_1 = f"batch/launch_generation_1.pth"
# MODEL_PATH_2 = f"batch/launch_generation_2.pth"
# MODEL_PATH_3 = f"{TYPE}/test_train_0.pth"
# MODEL_PATH_4 = f"{TYPE}/test_train_1.pth"
# MODEL_PATH_5 = f"{TYPE}/test_train_2.pth"
# MODEL_PATH_6 = f"{TYPE}/test_train_3.pth"

# MODEL_PATH_7 = f"{TYPE}/test_train_4.pth"
# MODEL_PATH_8 = f"{TYPE}/test_train_5.pth"
# MODEL_PATH_9 = f"{TYPE}/test_train_6.pth"
# MODEL_PATH_10 = f"{TYPE}/test_train_7.pth"

# MODEL_PATH_11 = f"{TYPE}/test_train_8.pth"
# MODEL_PATH_12 = f"{TYPE}/test_train_9.pth"
# MODEL_PATH_13 = f"{TYPE}/test_train_10.pth"

# MODEL_PATH_14 = f"{TYPE}/test_train_11.pth"
# MODEL_PATH_15 = f"{TYPE}/test_train_12.pth"
# MODEL_PATH_16 = f"{TYPE}/test_train_13.pth"
# MODEL_PATH_17 = f"{TYPE}/test_train_14.pth"

MODEL_PATH_18 = f"{TYPE}/test_train_15.pth"
MODEL_PATH_19 = f"{TYPE}/test_train_16.pth"
MODEL_PATH_20 = f"{TYPE}/test_train_17.pth"
MODEL_PATH_21 = f"{TYPE}/test_train_18.pth"
MODEL_PATH_22 = f"{TYPE}/test_train_19.pth"
MODEL_PATH_23 = f"{TYPE}/test_train_20.pth"
EPISODE = "launch_1"
batch_size = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

db = GoDatabaseLMDB(path= GAMES_DIR/"batch", db_name= EPISODE)
train_loader = get_data(db, batch_size)

# player_1 = GoBot()
# player_1.load_model(MODEL_DIR_9X9/MODEL_PATH_1)
# player_2 = GoBot()
# player_2.load_model(MODEL_DIR_9X9/MODEL_PATH_2)
# player_3 = GoBot()
# player_3.load_model(MODEL_DIR_9X9/MODEL_PATH_3)
# player_4 = GoBot()
# player_4.load_model(MODEL_DIR_9X9/MODEL_PATH_4)
# player_5 = GoBot()
# player_5.load_model(MODEL_DIR_9X9/MODEL_PATH_5)
# player_6 = GoBot()
# player_6.load_model(MODEL_DIR_9X9/MODEL_PATH_6)
# player_7 = GoBot()
# player_7.load_model(MODEL_DIR_9X9/MODEL_PATH_7)

# player_8 = GoBot()
# player_8.load_model(MODEL_DIR_9X9/MODEL_PATH_8)
# player_9 = GoBot()
# player_9.load_model(MODEL_DIR_9X9/MODEL_PATH_9)
# player_10 = GoBot()
# player_10.load_model(MODEL_DIR_9X9/MODEL_PATH_10)

# player_11 = GoBot()
# player_11.load_model(MODEL_DIR_9X9/MODEL_PATH_11)
# player_12 = GoBot()
# player_12.load_model(MODEL_DIR_9X9/MODEL_PATH_12)
# player_13 = GoBot()
# player_13.load_model(MODEL_DIR_9X9/MODEL_PATH_13)

# player_14 = GoBot()
# player_14.load_model(MODEL_DIR_9X9/MODEL_PATH_14)
# player_15 = GoBot()
# player_15.load_model(MODEL_DIR_9X9/MODEL_PATH_15)
# player_16 = GoBot()
# player_16.load_model(MODEL_DIR_9X9/MODEL_PATH_16)
# player_17 = GoBot()
# player_17.load_model(MODEL_DIR_9X9/MODEL_PATH_17)

player_18 = GoBot()
player_18.load_model(MODEL_DIR_9X9/MODEL_PATH_18)
player_19 = GoBot()
player_19.load_model(MODEL_DIR_9X9/MODEL_PATH_19)
player_20 = GoBot()
player_20.load_model(MODEL_DIR_9X9/MODEL_PATH_20)
player_21 = GoBot()
player_21.load_model(MODEL_DIR_9X9/MODEL_PATH_21)
player_22 = GoBot()
player_22.load_model(MODEL_DIR_9X9/MODEL_PATH_22)
player_23 = GoBot()
player_23.load_model(MODEL_DIR_9X9/MODEL_PATH_23)

# loss1 = test(player_1, train_loader, device=device)
# loss2 = test(player_2, train_loader, device=device)
# loss3 = test(player_3, train_loader, device=device)
# loss4 = test(player_4, train_loader, device=device)
# loss5 = test(player_5, train_loader, device=device)
# loss6 = test(player_6, train_loader, device=device)

# loss7 = test(player_7, train_loader, device=device)
# loss8 = test(player_8, train_loader, device=device)
# loss9 = test(player_9, train_loader, device=device)

# loss10 = test(player_10, train_loader, device=device)
# loss11 = test(player_11, train_loader, device=device)
# loss12 = test(player_12, train_loader, device=device)
# loss13 = test(player_13, train_loader, device=device)

# loss14 = test(player_14, train_loader, device=device)
# loss15 = test(player_15, train_loader, device=device)
# loss16 = test(player_16, train_loader, device=device)
# loss17 = test(player_17, train_loader, device=device)

loss18 = test(player_18, train_loader, device=device)
loss19 = test(player_19, train_loader, device=device)
loss20 = test(player_20, train_loader, device=device)
loss21 = test(player_21, train_loader, device=device)
loss22 = test(player_22, train_loader, device=device)
loss23 = test(player_23, train_loader, device=device)

# print(loss1)
# print(loss2)
# print(loss3)
# print(loss4)
# print(loss5)
# print(loss6)

# print(loss7)
# print(loss8)
# print(loss9)

# print(loss10)
# print(loss11)
# print(loss12)
# print(loss13)

# print(loss14)
# print(loss15)
# print(loss16)
# print(loss17)

print(loss18)
print(loss19)
print(loss20)
print(loss21)
print(loss22)
print(loss23)