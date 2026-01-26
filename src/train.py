
from RL_GoBot import var
from RL_GoBot.model import GoBot
from RL_GoBot.learning import train_one_episode
from RL_GoBot.data_base import GoDatabaseLMDB, get_data
from config import GAMES_DIR, MODEL_DIR_9X9

import torch

TYPE = "batch"
ID = 0
GENERATION = f"launch_generation_{ID}"       # a better name would be generation
EPISODE = f"launch_{ID}"

db = GoDatabaseLMDB(path = GAMES_DIR/TYPE, db_name= EPISODE)    # collection_name will be different for each of the episodes of the bot learning
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # test 1   -> 294.2 <-
# bot = GoBot()
# bot.load_model("{}/{}.pth".format(MODEL_DIR_9X9/TYPE, GENERATION))
# train_one_episode(bot, db, batch_size=32, device = device, learning_rate=0.001, temperature=0, weight_decay=0, epochs=20)
# bot.save_model("{}/test_train_{}.pth".format(MODEL_DIR_9X9/TYPE, 0))

# # test 2   -> 370.6 <-
# bot = GoBot()
# bot.load_model("{}/{}.pth".format(MODEL_DIR_9X9/TYPE, GENERATION))
# train_one_episode(bot, db, batch_size=64, device = device, learning_rate=0.001, temperature=0, weight_decay=0, epochs=20)
# bot.save_model("{}/test_train_{}.pth".format(MODEL_DIR_9X9/TYPE, 1))

# # test 3   -> 239.1 <-
# bot = GoBot()
# bot.load_model("{}/{}.pth".format(MODEL_DIR_9X9/TYPE, GENERATION))
# train_one_episode(bot, db, batch_size=32, device = device, learning_rate=0.01, temperature=0, weight_decay=0, epochs=20)
# bot.save_model("{}/test_train_{}.pth".format(MODEL_DIR_9X9/TYPE, 2))

# # test 4   -> 254.7 <-
# bot = GoBot()
# bot.load_model("{}/{}.pth".format(MODEL_DIR_9X9/TYPE, GENERATION))
# train_one_episode(bot, db, batch_size=32, device = device, learning_rate=0.001, temperature=0, weight_decay=0, epochs=30)
# bot.save_model("{}/test_train_{}.pth".format(MODEL_DIR_9X9/TYPE, 3))


# # test 5  -> 242.0 <-
# bot = GoBot()
# bot.load_model("{}/{}.pth".format(MODEL_DIR_9X9/TYPE, GENERATION))
# train_one_episode(bot, db, batch_size=32, device = device, learning_rate=0.005, temperature=0, weight_decay=0, epochs=20)
# bot.save_model("{}/test_train_{}.pth".format(MODEL_DIR_9X9/TYPE, 4))

# # test 6   -> 241.9 <-
# bot = GoBot()
# bot.load_model("{}/{}.pth".format(MODEL_DIR_9X9/TYPE, GENERATION))
# train_one_episode(bot, db, batch_size=32, device = device, learning_rate=0.005, temperature=0, weight_decay=0.0003, epochs=20)
# bot.save_model("{}/test_train_{}.pth".format(MODEL_DIR_9X9/TYPE, 5))

# # test 7   -> 242.4 <-
# bot = GoBot()
# bot.load_model("{}/{}.pth".format(MODEL_DIR_9X9/TYPE, GENERATION))
# train_one_episode(bot, db, batch_size=32, device = device, learning_rate=0.005, temperature=0.005, weight_decay=0, epochs=20)
# bot.save_model("{}/test_train_{}.pth".format(MODEL_DIR_9X9/TYPE, 6))


# # test 8   -> 242.5 <-
# bot = GoBot()
# bot.load_model("{}/{}.pth".format(MODEL_DIR_9X9/TYPE, GENERATION))
# train_one_episode(bot, db, batch_size=32, device = device, learning_rate=0.005, temperature=0.01, weight_decay=0, epochs=20)
# bot.save_model("{}/test_train_{}.pth".format(MODEL_DIR_9X9/TYPE, 7))

# # test 9   -> 242.4 <-
# bot = GoBot()
# bot.load_model("{}/{}.pth".format(MODEL_DIR_9X9/TYPE, GENERATION))
# train_one_episode(bot, db, batch_size=32, device = device, learning_rate=0.005, temperature=0.01, weight_decay=0.0003, epochs=20)
# bot.save_model("{}/test_train_{}.pth".format(MODEL_DIR_9X9/TYPE, 8))

# # test 10   -> 355.6 <-
# bot = GoBot()
# bot.load_model("{}/{}.pth".format(MODEL_DIR_9X9/TYPE, GENERATION))
# train_one_episode(bot, db, batch_size=64, device = device, learning_rate=0.001, temperature=0, weight_decay=0, epochs=30)
# bot.save_model("{}/test_train_{}.pth".format(MODEL_DIR_9X9/TYPE, 9))

# # test 11   -> 239.4 <-
# bot = GoBot()
# bot.load_model("{}/{}.pth".format(MODEL_DIR_9X9/TYPE, GENERATION))
# train_one_episode(bot, db, batch_size=64, device = device, learning_rate=0.01, temperature=0, weight_decay=0, epochs=30)
# bot.save_model("{}/test_train_{}.pth".format(MODEL_DIR_9X9/TYPE, 10))


# # test12   -> 239.8 <-
# bot = GoBot()
# bot.load_model("{}/{}.pth".format(MODEL_DIR_9X9/TYPE, GENERATION))
# train_one_episode(bot, db, batch_size=64, device = device, learning_rate=0.01, temperature=0.01, weight_decay=0, epochs=30)
# bot.save_model("{}/test_train_{}.pth".format(MODEL_DIR_9X9/TYPE, 11))

# # test13   -> 244.0 <-
# bot = GoBot()
# bot.load_model("{}/{}.pth".format(MODEL_DIR_9X9/TYPE, GENERATION))
# train_one_episode(bot, db, batch_size=64, device = device, learning_rate=0.01, temperature=0.03, weight_decay=0, epochs=30)
# bot.save_model("{}/test_train_{}.pth".format(MODEL_DIR_9X9/TYPE, 12))

# # test14   -> 242.0 <-
# bot = GoBot()
# bot.load_model("{}/{}.pth".format(MODEL_DIR_9X9/TYPE, GENERATION))
# train_one_episode(bot, db, batch_size=64, device = device, learning_rate=0.01, temperature=0, weight_decay=0.003, epochs=30)
# bot.save_model("{}/test_train_{}.pth".format(MODEL_DIR_9X9/TYPE, 13))

# # test15   -> 247.2 <-
# bot = GoBot()
# bot.load_model("{}/{}.pth".format(MODEL_DIR_9X9/TYPE, GENERATION))
# train_one_episode(bot, db, batch_size=64, device = device, learning_rate=0.01, temperature=0, weight_decay=0.01, epochs=30)
# bot.save_model("{}/test_train_{}.pth".format(MODEL_DIR_9X9/TYPE, 14))


# # test12   -> 248.5 <-
# bot = GoBot()
# bot.load_model("{}/{}.pth".format(MODEL_DIR_9X9/TYPE, GENERATION))
# train_one_episode(bot, db, batch_size=64, device = device, learning_rate=0.01, temperature=0.01, weight_decay=0.01, epochs=20)
# bot.save_model("{}/test_train_{}.pth".format(MODEL_DIR_9X9/TYPE, 15))

# # test12   -> 246.7 <- 
# bot = GoBot()
# bot.load_model("{}/{}.pth".format(MODEL_DIR_9X9/TYPE, GENERATION))
# train_one_episode(bot, db, batch_size=64, device = device, learning_rate=0.01, temperature=0.01, weight_decay=0.01, epochs=30)
# bot.save_model("{}/test_train_{}.pth".format(MODEL_DIR_9X9/TYPE, 16))

# # test12   -> 246.2 <-
# bot = GoBot()
# bot.load_model("{}/{}.pth".format(MODEL_DIR_9X9/TYPE, GENERATION))
# train_one_episode(bot, db, batch_size=64, device = device, learning_rate=0.01, temperature=0.01, weight_decay=0.01, epochs=40)
# bot.save_model("{}/test_train_{}.pth".format(MODEL_DIR_9X9/TYPE, 17))

# # test12   -> 249.3 <-
# bot = GoBot()
# bot.load_model("{}/{}.pth".format(MODEL_DIR_9X9/TYPE, GENERATION))
# train_one_episode(bot, db, batch_size=64, device = device, learning_rate=0.03, temperature=0.01, weight_decay=0.01, epochs=30)
# bot.save_model("{}/test_train_{}.pth".format(MODEL_DIR_9X9/TYPE, 18))

# # test12   -> 270.9 <-
# bot = GoBot()
# bot.load_model("{}/{}.pth".format(MODEL_DIR_9X9/TYPE, GENERATION))
# train_one_episode(bot, db, batch_size=64, device = device, learning_rate=0.1, temperature=0.01, weight_decay=0.01, epochs=30)
# bot.save_model("{}/test_train_{}.pth".format(MODEL_DIR_9X9/TYPE, 19))


# # Final   -> 253.7 <-
# bot = GoBot()
# bot.load_model("{}/{}.pth".format(MODEL_DIR_9X9/TYPE, GENERATION))
# train_one_episode(bot, db, batch_size=64, device = device, learning_rate=0.01, temperature=0.03, weight_decay=0.01, epochs=20)
# bot.save_model("{}/launch_generation_{}.pth".format(MODEL_DIR_9X9/TYPE, 1))




