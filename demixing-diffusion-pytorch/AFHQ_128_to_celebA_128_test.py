#from comet_ml import Experiment
from demixing_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import torchvision
import os
import errno
import shutil
import argparse

def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

def del_folder(path):
    try:
        shutil.rmtree(path)
    except OSError as exc:
        pass


parser = argparse.ArgumentParser()
parser.add_argument('--time_steps', default=50, type=int)
parser.add_argument('--train_steps', default=700000, type=int)
parser.add_argument('--save_folder', default='./results_cifar10', type=str)
parser.add_argument('--data_path_start', default='../deblurring-diffusion-pytorch/AFHQ/afhq/train/', type=str)
parser.add_argument('--data_path_end', default='../deblurring-diffusion-pytorch/root_celebA_128_train_new/', type=str)
parser.add_argument('--load_path', default=None, type=str)
parser.add_argument('--train_routine', default='Final', type=str)
parser.add_argument('--sampling_routine', default='default', type=str)
parser.add_argument('--remove_time_embed', action="store_true")
parser.add_argument('--residual', action="store_true")
parser.add_argument('--loss_type', default='l1', type=str)
parser.add_argument('--test_type', default='train_data', type=str)
parser.add_argument('--noise', default=0, type=float)

# set gpu list
parser.add_argument('--gpu_list', default='0', type=str)

args = parser.parse_args()
print(args)

# set gpu list
gpu_list = args.gpu_list
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img_path=None
if 'train' in args.test_type:
    img_path = args.data_path_start
elif 'test' in args.test_type:
    img_path = args.data_path_start


model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels=3,
    with_time_emb=not(args.remove_time_embed),
    residual=args.residual
).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    channels = 3,
    timesteps = args.time_steps,   # number of steps
    loss_type = args.loss_type,    # L1 or L2
    train_routine = args.train_routine,
    sampling_routine = args.sampling_routine
).cuda()

import torch
diffusion = torch.nn.DataParallel(diffusion, device_ids=range(torch.cuda.device_count()))


trainer = Trainer(
    diffusion,
    args.data_path_end,
    img_path,
    image_size = 128,
    train_batch_size = 32,
    train_lr = 2e-5,
    train_num_steps = args.train_steps,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    fp16 = False,                       # turn on mixed precision training with apex
    results_folder = args.save_folder,
    load_path = args.load_path,
    dataset = 'train'
)

if args.test_type == 'train_data':
    trainer.test_from_data('train', s_times=None)

elif args.test_type == 'test_data':
    trainer.test_from_data('test', s_times=None)


#### for FID and noise ablation ##
elif args.test_type == 'test_sample_and_save_for_fid':
    trainer.sample_and_save_for_fid(args.noise)

########## for paper ##########

elif args.test_type == 'train_paper_showing_diffusion_images_cover_page':
    trainer.paper_showing_diffusion_images_cover_page()

elif args.test_type == 'test_paper_showing_diffusion_images_cover_page':
    trainer.paper_showing_diffusion_images_cover_page()