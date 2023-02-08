import torch
import torch.nn as nn
from pytorch_Part import training_base
from ParT import ParticleTransformer, ParticleTransformerTrim
#from pytorch_deepjet_transformer_v4_corr import DeepJetTransformer, ParticleTransformer
#from ParT_old import DeepJetTransformer, ParticleTransformer
from pytorch_ranger import *
#from schedulers import *
import wandb

wandb_params = {
    "num_epochs": 50,
    "lr_mult": 0.3,
    "lr_rate": 0.005,
    "num_enc": 3,
    "num_head": 8,
    "embed_dim": 128,
    "BS": 2048,
}
wandb.init(project="StudyParticleTransformer", tags=["CMS"],
           resume=False,
           entity="munozariasjm",
           config={c: k for c, k in wandb_params.items() if isinstance(k, (int, float, str))},
           )

def cross_entropy_one_hot(input, target):
    _, labels = target.max(dim=1)
    return nn.CrossEntropyLoss()(input, labels)

num_epochs = wandb_params["num_epochs"]


lr_epochs = max(1, int(num_epochs * wandb_params["lr_mult"]))
lr_rate = wandb_params["lr_rate"] ** (1.0 / lr_epochs)
mil = list(range(num_epochs - lr_epochs, num_epochs))
print(lr_rate)
print(mil)

# Dropping all vtx info

model = ParticleTransformerTrim(num_classes = 6,
                            num_enc = wandb_params["num_enc"],
                            num_head = wandb_params["num_head"],
                            embed_dim = wandb_params["embed_dim"],
                            cpf_dim = 29, ## TODO: change to 17
                            npf_dim = 8,
                            vtx_dim = 14,
                            for_inference = False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
wandb.watch(model, log='all')

#model = nn.DataParallel(model)
scaler = torch.cuda.amp.GradScaler()

criterion = cross_entropy_one_hot
optimizer = Ranger(model.parameters(), lr = 1e-3)
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [1], gamma = 0.1)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = mil, gamma = lr_rate)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=14714, T_mult=1, eta_min=5e-6, last_epoch=-1)
#scheduler = CosineAnealingWarmRestartsWeightDecay(optimizer, T_0=14714, T_mul=1, eta_min=5e-6, last_epoch=-1, gamma = 0.9, max_lr = 5e-4)

train=training_base(model = model,
                    criterion = criterion,
                    optimizer = optimizer,
                    scheduler = scheduler,
                    scaler = scaler,
                    testrun=False,
                    logger = wandb,
                    )

train.train_data.maxFilesOpen=1

#model,history = train.trainModel(nepochs=num_epochs, batchsize=wandb_params["BS"])
train.trainModel(nepochs=num_epochs, batchsize=wandb_params["BS"]);