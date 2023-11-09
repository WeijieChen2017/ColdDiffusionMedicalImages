import os
import numpy as np
import argparse
import random
from torch.utils import data
from demixing_diffusion_pytorch import Unet, DatasetPaired_Aug, cycle
from demixing_diffusion_pytorch import period_trainer_MR2CT as trainer

parser = argparse.ArgumentParser()
parser.add_argument('--time_steps', default=1000, type=int)
parser.add_argument('--n_jumps', default=1, type=int)
parser.add_argument('--n_test', default=-1, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--train_epochs', default=700000, type=int)
parser.add_argument('--save_folder', default='./results/', type=str)
parser.add_argument('--load_path', default='./proj/MR2CT_simpleUNet/model_600000.pt', type=str)
parser.add_argument('--data_path', default='./data/MR2CT/MR_x_2d/', type=str)
parser.add_argument('--time_emb', action="store_true")
parser.add_argument('--residual', action="store_true")
parser.add_argument('--loss_type', default='l1', type=str)
parser.add_argument('--gpu_list', default='0', type=str)
parser.add_argument('--seed', default=426, type=int)



args = parser.parse_args()
print(args)

# set gpu list
gpu_list = args.gpu_list
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



SEED = args.seed
# Set the random seed for all the libraries
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels=3,
    with_time_emb=not(args.time_emb),
    residual=not(args.residual),
)

# load the model
# model.load_state_dict(torch.load(args.model_path))
# model.cuda()
# model.eval()

dataloader = data.DataLoader(
    DatasetPaired_Aug(
        folder = args.data_path,
        image_size = 256,
        stage='test',)
    , 
    batch_size = args.batch_size, 
    shuffle="False", 
    pin_memory=True, 
    num_workers=16, 
    drop_last=False)

trainer = trainer(
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

model_type = args.load_path.split('/')[-2]
args.save_folder = args.save_folder + model_type + "/"

# create folder if not exisit
if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)

# trainer.load(args.load_path) # this is automatically done in the trainer class
trainer.eval_jumps(args.n_jumps, args.time_steps, args.n_test, args.save_folder)




