
from RL_GoBot import var
from RL_GoBot.model import GoBot
from RL_GoBot.learning import train_one_episode
from RL_GoBot.data_base import GoDatabaseLMDB, get_data
from config import GAMES_DIR, MODEL_DIR_9X9

import torch

ORIGINE_TYPE = "batch"
DESTINATION_TYPE = "test_gen_2"
ID = 1
GENERATION = f"launch_generation_{ID}"       # a better name would be generation
EPISODE = f"launch_{ID}"

db = GoDatabaseLMDB(path = GAMES_DIR/ORIGINE_TYPE, db_name= EPISODE)    # collection_name will be different for each of the episodes of the bot learning
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # test 4   -> 121 <-
# bot = GoBot()
# bot.load_model("{}/{}.pth".format(MODEL_DIR_9X9/ORIGINE_TYPE, GENERATION))
# train_one_episode(bot, db, batch_size=32, device = device, learning_rate=0.001, temperature=0, weight_decay=0, epochs=20)
# bot.save_model("{}/test_train_{}.pth".format(MODEL_DIR_9X9/DESTINATION_TYPE, 0))

# # test 4   -> 115 <-
# bot = GoBot()
# bot.load_model("{}/{}.pth".format(MODEL_DIR_9X9/ORIGINE_TYPE, GENERATION))
# train_one_episode(bot, db, batch_size=32, device = device, learning_rate=0.003, temperature=0, weight_decay=0, epochs=20)
# bot.save_model("{}/test_train_{}.pth".format(MODEL_DIR_9X9/DESTINATION_TYPE, 1))

# # test 3   -> 111 <-
# bot = GoBot()
# bot.load_model("{}/{}.pth".format(MODEL_DIR_9X9/ORIGINE_TYPE, GENERATION))
# train_one_episode(bot, db, batch_size=32, device = device, learning_rate=0.01, temperature=0, weight_decay=0, epochs=20)
# bot.save_model("{}/test_train_{}.pth".format(MODEL_DIR_9X9/DESTINATION_TYPE, 2))

# # test 3   -> 97 <-
# bot = GoBot()
# bot.load_model("{}/{}.pth".format(MODEL_DIR_9X9/ORIGINE_TYPE, GENERATION))
# train_one_episode(bot, db, batch_size=32, device = device, learning_rate=0.03, temperature=0, weight_decay=0, epochs=20)
# bot.save_model("{}/test_train_{}.pth".format(MODEL_DIR_9X9/DESTINATION_TYPE, 3))



# # test 4   -> 113 <-
# bot = GoBot()
# bot.load_model("{}/{}.pth".format(MODEL_DIR_9X9/ORIGINE_TYPE, GENERATION))
# train_one_episode(bot, db, batch_size=64, device = device, learning_rate=0.01, temperature=0, weight_decay=0, epochs=20)
# bot.save_model("{}/test_train_{}.pth".format(MODEL_DIR_9X9/DESTINATION_TYPE, 4))

# # test 4   -> 115 <-
# bot = GoBot()
# bot.load_model("{}/{}.pth".format(MODEL_DIR_9X9/ORIGINE_TYPE, GENERATION))
# train_one_episode(bot, db, batch_size=128, device = device, learning_rate=0.01, temperature=0, weight_decay=0, epochs=20)
# bot.save_model("{}/test_train_{}.pth".format(MODEL_DIR_9X9/DESTINATION_TYPE, 5))



# # test 3   -> 110 <-    # vraiment inutile
# bot = GoBot()
# bot.load_model("{}/{}.pth".format(MODEL_DIR_9X9/ORIGINE_TYPE, GENERATION))
# train_one_episode(bot, db, batch_size=32, device = device, learning_rate=0.01, temperature=0, weight_decay=0, epochs=30)
# bot.save_model("{}/test_train_{}.pth".format(MODEL_DIR_9X9/DESTINATION_TYPE, 6))

# # test 3   -> 111 <-
# bot = GoBot()
# bot.load_model("{}/{}.pth".format(MODEL_DIR_9X9/ORIGINE_TYPE, GENERATION))
# train_one_episode(bot, db, batch_size=64, device = device, learning_rate=0.01, temperature=0, weight_decay=0, epochs=30)
# bot.save_model("{}/test_train_{}.pth".format(MODEL_DIR_9X9/DESTINATION_TYPE, 7))

#####

# # test 5  -> 112.4 <-
# bot = GoBot()
# bot.load_model("{}/{}.pth".format(MODEL_DIR_9X9/ORIGINE_TYPE, GENERATION))
# train_one_episode(bot, db, batch_size=64, device = device, learning_rate=0.02, temperature=0, weight_decay=0, epochs=20)
# bot.save_model("{}/test_train_{}.pth".format(MODEL_DIR_9X9/DESTINATION_TYPE, 8))

# # test 5  -> 113.6 <-
# bot = GoBot()
# bot.load_model("{}/{}.pth".format(MODEL_DIR_9X9/ORIGINE_TYPE, GENERATION))
# train_one_episode(bot, db, batch_size=64, device = device, learning_rate=0.02, temperature=0, weight_decay=0.003, epochs=20)
# bot.save_model("{}/test_train_{}.pth".format(MODEL_DIR_9X9/DESTINATION_TYPE, 9))

# # test 6   -> 118.4 <-
# bot = GoBot()
# bot.load_model("{}/{}.pth".format(MODEL_DIR_9X9/ORIGINE_TYPE, GENERATION))
# train_one_episode(bot, db, batch_size=64, device = device, learning_rate=0.02, temperature=0, weight_decay=0.01, epochs=20)
# bot.save_model("{}/test_train_{}.pth".format(MODEL_DIR_9X9/DESTINATION_TYPE, 10))

# # test 7   -> 129.3 <-
# bot = GoBot()
# bot.load_model("{}/{}.pth".format(MODEL_DIR_9X9/ORIGINE_TYPE, GENERATION))
# train_one_episode(bot, db, batch_size=64, device = device, learning_rate=0.02, temperature=0, weight_decay=0.03, epochs=20)
# bot.save_model("{}/test_train_{}.pth".format(MODEL_DIR_9X9/DESTINATION_TYPE, 11))


# # test 8   -> 111.4 <-
# bot = GoBot()
# bot.load_model("{}/{}.pth".format(MODEL_DIR_9X9/ORIGINE_TYPE, GENERATION))
# train_one_episode(bot, db, batch_size=64, device = device, learning_rate=0.02, temperature=0.003, weight_decay=0, epochs=20)
# bot.save_model("{}/test_train_{}.pth".format(MODEL_DIR_9X9/DESTINATION_TYPE, 12))

# # test 9   -> 112.4 <-
# bot = GoBot()
# bot.load_model("{}/{}.pth".format(MODEL_DIR_9X9/ORIGINE_TYPE, GENERATION))
# train_one_episode(bot, db, batch_size=64, device = device, learning_rate=0.02, temperature=0.01, weight_decay=0, epochs=20)
# bot.save_model("{}/test_train_{}.pth".format(MODEL_DIR_9X9/DESTINATION_TYPE, 13))

# # test 10   -> 125.7 <-
# bot = GoBot()
# bot.load_model("{}/{}.pth".format(MODEL_DIR_9X9/ORIGINE_TYPE, GENERATION))
# train_one_episode(bot, db, batch_size=64, device = device, learning_rate=0.02, temperature=0.03, weight_decay=0, epochs=20)
# bot.save_model("{}/test_train_{}.pth".format(MODEL_DIR_9X9/DESTINATION_TYPE, 14))

###

# # test 11   -> 118 <-
# bot = GoBot()
# bot.load_model("{}/{}.pth".format(MODEL_DIR_9X9/ORIGINE_TYPE, GENERATION))
# train_one_episode(bot, db, batch_size=64, device = device, learning_rate=0.003, temperature=0, weight_decay=0, epochs=20)
# bot.save_model("{}/test_train_{}.pth".format(MODEL_DIR_9X9/DESTINATION_TYPE, 15))

# # test12   -> 113 <-
# bot = GoBot()
# bot.load_model("{}/{}.pth".format(MODEL_DIR_9X9/ORIGINE_TYPE, GENERATION))
# train_one_episode(bot, db, batch_size=64, device = device, learning_rate=0.01, temperature=0, weight_decay=0, epochs=20)
# bot.save_model("{}/test_train_{}.pth".format(MODEL_DIR_9X9/DESTINATION_TYPE, 16))

# # test13   -> 112 <-
# bot = GoBot()
# bot.load_model("{}/{}.pth".format(MODEL_DIR_9X9/ORIGINE_TYPE, GENERATION))
# train_one_episode(bot, db, batch_size=64, device = device, learning_rate=0.03, temperature=0, weight_decay=0, epochs=20)
# bot.save_model("{}/test_train_{}.pth".format(MODEL_DIR_9X9/DESTINATION_TYPE, 17))


# # test 11   -> 116 <-
# bot = GoBot()
# bot.load_model("{}/{}.pth".format(MODEL_DIR_9X9/ORIGINE_TYPE, GENERATION))
# train_one_episode(bot, db, batch_size=64, device = device, learning_rate=0.003, temperature=0, weight_decay=0, epochs=30)
# bot.save_model("{}/test_train_{}.pth".format(MODEL_DIR_9X9/DESTINATION_TYPE, 18))

# # test12   -> 111 <-
# bot = GoBot()
# bot.load_model("{}/{}.pth".format(MODEL_DIR_9X9/ORIGINE_TYPE, GENERATION))
# train_one_episode(bot, db, batch_size=64, device = device, learning_rate=0.01, temperature=0, weight_decay=0, epochs=30)
# bot.save_model("{}/test_train_{}.pth".format(MODEL_DIR_9X9/DESTINATION_TYPE, 19))

# # test13   -> 103 <-
# bot = GoBot()
# bot.load_model("{}/{}.pth".format(MODEL_DIR_9X9/ORIGINE_TYPE, GENERATION))
# train_one_episode(bot, db, batch_size=64, device = device, learning_rate=0.03, temperature=0, weight_decay=0, epochs=30)
# bot.save_model("{}/test_train_{}.pth".format(MODEL_DIR_9X9/DESTINATION_TYPE, 20))

