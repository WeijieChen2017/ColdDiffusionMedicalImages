import os
import numpy as np
import argparse
from torch.utils import data
from demixing_diffusion_pytorch import Unet, DatasetPaired_Aug, simple_trainer, cycle

parser = argparse.ArgumentParser()
parser.add_argument('--time_steps', default=1000, type=int)
parser.add_argument('--train_epochs', default=700000, type=int)
parser.add_argument('--batch_size', default=8, type=int)   
parser.add_argument('--save_folder', default='./proj/MR2CT_simpleUNet/', type=str)
parser.add_argument('--data_path', default='./data/MR2CT/MR_x_2d/', type=str)
parser.add_argument('--load_path', default=None, type=str)
parser.add_argument('--time_emb', action="store_true")
parser.add_argument('--residual', action="store_true")
parser.add_argument('--loss_type', default='l1', type=str)
parser.add_argument('--gpu_list', default='3', type=str)


args = parser.parse_args()
print(args)

# set gpu list
gpu_list = args.gpu_list
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels=3,
    with_time_emb=not(args.time_emb),
    residual=not(args.residual),
).cuda()

dataloader = cycle(data.DataLoader(
    DatasetPaired_Aug(
        folder = args.data_path,
        image_size = 256,)
    , 
    batch_size = args.batch_size, 
    shuffle="True", 
    pin_memory=True, 
    num_workers=16, 
    drop_last=True)
)

trainer = simple_trainer(
    model,
    dataloader,
    time_steps = args.time_steps,
    loss_type = args.loss_type,
    image_size = 256,
    train_batch_size = args.batch_size,
    train_lr = 2e-5,
    train_num_steps = args.train_epochs,         # total training steps
    gradient_accumulate_every = 1,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    fp16 = False,                       # turn on mixed precision training with apex
    results_folder = args.save_folder,
    load_path = args.load_path,
)

trainer.train()
