import torch
import torch.nn as nn
import wandb
from pytorch_first_try import training_base
from pytorch_deepjet import *
from pytorch_deepjet_transformer import DeepJetTransformer
from pytorch_ranger import Ranger

def cross_entropy_one_hot(input, target):
    _, labels = target.max(dim=1)
    return nn.CrossEntropyLoss()(input, labels)

def select_model(model_name):
    if model_name == 'DeepJet':
        return DeepJet(num_classes = 6)
    elif model_name == 'DeepJetTransformer':
        return DeepJetTransformer(num_classes = 6)
    else:
        raise ValueError('Model name not recognized')


config_d = {
    "num_epochs": 50,
    "model_name": 'DeepJet',
    "batchsize"=1024*4,
    "optimizer": "Ranger"
}



if __name__ == '__main__':

    wandb.init(project='DeepJetCERN', config=config_d, entity='munozariasjm')

    lr_epochs = max(1, int(config_d["num_epochs"] * 0.3))
    lr_rate = 0.02 ** (1.0 / lr_epochs)
    mil = list(range(config_d["num_epochs"] - lr_epochs, config_d["num_epochs"]))
    model = select_model(config_d["model_name"])
    wandb.watch(model, log='all')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = cross_entropy_one_hot
    if config_d["optimizer"] == "Ranger":
        optimizer = Ranger(model.parameters(), lr = 5e-3)
    optimizer = Ranger(model.parameters(), lr = 5e-3)
    elif config_d["optimizer"] == "Adam":
        torch.optim.Adam(model.parameters(), lr = 0.003, eps = 1e-07)
    elif config_d["optimizer"] == "AdamW":
        torch.optim.AdamW(model.parameters(), lr = 0.003, eps = 1e-07)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = mil, gamma = lr_rate)
    train=training_base(model = model, criterion = criterion, optimizer = optimizer, scheduler = scheduler, testrun=False, wb_logger=wandb)

    train.train_data.maxFilesOpen=1

    history = train.trainModel(nepochs=config_d["num_epochs"]+lr_epochs,
                                    config_d["batchsize"]=1024*4)

    wandb.finish()