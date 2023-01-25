# This code is used to finetune the IPT on the EUVP underwater dataset

import numpy as np
import math
import sys
import matplotlib.pyplot as plt

import torch
import glob
import utility_pretrain
import data
import loss
import warnings
warnings.filterwarnings('ignore')
import os
os.system('pip install einops')

import model
from option import args

torch.manual_seed(args.seed)

from torch.utils.data import ConcatDataset, DataLoader

from trainer_pretrain import Trainer
from training_utils import set_requires_grad, generate_data 
from training_utils import EuVPDataset


# Configuration Changes
checkpoint = utility_pretrain.checkpoint(args)

checkpoint.args.test_only = False
checkpoint.args.dir_data = "CBSD68"    # This does not matter
checkpoint.args.pretrain = "pretrained_models/IPT_denoise30.pt"
#checkpoint.args.pretrain = "finetuned_models/model_e50.pth"
#checkpoint.args.pretrain = "pretrained_models/IPT_pretrain.pt"
checkpoint.args.data_test = ["CBSD68"]    # This does not matter as well
checkpoint.args.scale = [1]
checkpoint.args.denoise = True
checkpoint.args.cpu = False
checkpoint.args.n_GPUs = 1

if 'IPT_pretrain' in args.pretrain:
    args.num_queries = 6

data_folders = ["underwater_dark", "underwater_imagenet", "underwater_scenes"]


# 1st: Loading and understanding the IPT model.
base_model = model.Model(args, checkpoint)
base_model.model.load_state_dict(torch.load(args.pretrain), strict=False)
base_model.n_GPUs = checkpoint.args.n_GPUs
_loss = loss.Loss(args, checkpoint) if not args.test_only else None

ipt = base_model.model
head_params = set_requires_grad(ipt.head, requires_grad=False)
body_params = set_requires_grad(ipt.body, requires_grad=False)
tail_params = set_requires_grad(ipt.tail, requires_grad=False)

total_params = head_params + body_params + tail_params

print(f"IPT Head Parameters = {head_params:,}")
print(f"IPT Body Parameters = {body_params:,} (Vision Transformer)")
print(f"IPT Tail Parameters = {tail_params:,}")
print("================")
print(f"IPT Total Parameters = {total_params:,}")

_ = set_requires_grad(ipt, requires_grad=True)


# Finetuning hyperparameters such as lr, epochs, scheduler, etc.
lr_start, lr_stop = 2e-5, 2e-5
epochs, batch_size = 400, 8
save_every = 2
lr_change_every = 50
start_from = 0

# 2nd: Create and save EUVP cropped IPT input-output (48*48) pairs
print("\n\nGenerating EUVP Training Data")
generate_data(data_folders)
print('Data Generated')

# Datasets and the dataloader
IMG_SIZE = 48 
dark_dataset = EuVPDataset(f"train_data/EUVP/{data_folders[0]}", "input", "output")
imagenet_dataset = EuVPDataset(f"train_data/EUVP/{data_folders[1]}", "input", "output")
scenes_dataset = EuVPDataset(f"train_data/EUVP/{data_folders[2]}", "input", "output")

#dark_dataset = EuVPDataset(f"benchmark/EUVP/Paired/{data_folders[0]}", "trainA", "trainB", img_size=IMG_SIZE)
#imagenet_dataset = EuVPDataset(f"benchmark/EUVP/Paired/{data_folders[1]}", "trainA", "trainB", img_size=IMG_SIZE)
#scenes_dataset = EuVPDataset(f"benchmark/EUVP/Paired/{data_folders[2]}", "trainA", "trainB", img_size=IMG_SIZE)

train_datasets = ConcatDataset([dark_dataset, imagenet_dataset, scenes_dataset])
#train_datasets = ConcatDataset([imagenet_dataset, scenes_dataset])
train_loader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True, 
                          pin_memory=True, num_workers=4)

# define loss, optimizer and lr_scheduler
#ipt_loss = torch.nn.L1Loss()
ipt_loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(ipt.parameters(), lr=lr_start)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

def is_valid_number(x):
    return not(math.isnan(x) or math.isinf(x) or x > 1e4)

# Finetuning the IPT (Code writing and running).
losses = np.zeros((epochs,))
for epoch in range(start_from, epochs):
    epoch_losses = []
        
    for step, batch in enumerate(train_loader):
        outputs = ipt(batch[0].cuda())
        loss = ipt_loss(outputs, batch[1].cuda())

        if is_valid_number(loss.data.item()):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        val = len(train_loader) if len(train_loader) < 100 else 100 
        if step % int(len(train_loader)/val) == 0:
            print('Epoch: [{}/{}], Step: [{}/{}]: Loss: {:.3f}'.format(
                        epoch, epochs, step + 1, len(train_loader), loss))
            sys.stdout.flush()

    # Update learning rate 
    if epoch < epochs-1 and (epoch+1)%lr_change_every==0:   
        lr_scheduler.step()

    epoch_losses.append(loss.cpu().detach().numpy())
    losses[epoch] = np.mean(np.array(epoch_losses, dtype=np.float16))
    
    # save checkpoint
    net_path = os.path.join("finetuned_models", 'model_e%d.pth' % (epoch + 1))

    if (epoch+1)%save_every==0:
        torch.save(ipt.state_dict(), net_path)

plt.figure()
plt.plot(losses)
plt.title("IPT Finetuning Loss (EUVP)")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig("ipt_finetuning_loss.png")
    