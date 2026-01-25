
from RL_GoBot import var
from RL_GoBot.model import GoBot
from RL_GoBot.learning import get_optimizer, compute_loss, train
from RL_GoBot.data_base import GoDatabaseMongo, get_data

import torch

DATABASE = "batch"
EPISODE = 2
COLLECTION = f"episode_{EPISODE}"       # a better name would be generation

def train_one_episode(net : GoBot,
                      db : GoDatabaseMongo,
                      batch_size=var.BATCH_SIZE,
                      device='cuda',
                      learning_rate=var.LEARNING_RATE,
                      weight_decay=var.L2_LOSS,
                      momentum=var.MOMENTUM,
                      epochs=var.EPOCHS):
    '''
    Input arguments
    batch_size: Size of a mini-batch
    device: GPU where you want to train your network
    weight_decay: Weight decay co-efficient for regularization of weights
    momentum: Momentum for SGD optimizer
    epochs: Number of epochs for training the network
    '''
  
    optimizer = get_optimizer(net, learning_rate, weight_decay, momentum)

    for e in range(epochs):
        print("epoch : ", e)
        train_loader = get_data(db, batch_size)
        train(net, train_loader, optimizer, device)


bot = GoBot()
bot.load_model("{}/{}.pth".format(DATABASE, COLLECTION))

db = GoDatabaseMongo(db_name= DATABASE, collection_name = COLLECTION)    # collection_name will be different for each of the episodes of the bot learning
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_one_episode(bot, db, device = device)

bot.save_model("{}/episode_{}.pth".format(DATABASE, EPISODE+1))


