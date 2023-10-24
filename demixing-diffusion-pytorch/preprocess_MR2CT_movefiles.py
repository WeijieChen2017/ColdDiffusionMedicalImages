# we will receive the total numbers and the ratio of train/val/test

import os
import numpy as np
import nibabel as nib
import argparse
import glob
import tqdm
import json

from util import prepare_folder

parser = argparse.ArgumentParser()
parser.add_argument('--x_path', default='./data/MR2CT/MR', type=str)
parser.add_argument('--y_path', default='./data/MR2CT/CT', type=str)
parser.add_argument('--train_ratio', default=0.7, type=float)
parser.add_argument('--val_ratio', default=0.2, type=float)
parser.add_argument('--test_ratio', default=0.1, type=float)
parser.add_argument('--seed', default=1024, type=int)
parser.add_argument('--x_save_folder', default='./data/MR2CT/MR_2d', type=str)
parser.add_argument('--y_save_folder', default='./data/MR2CT/CT_2d', type=str)
parser.add_argument('--num_files', default=100, type=int)
args = parser.parse_args()

# load the parameters
x_path = args.x_path
y_path = args.y_path
train_ratio = args.train_ratio
val_ratio = args.val_ratio
test_ratio = args.test_ratio
seed = args.seed
x_save_folder = args.x_save_folder
y_save_folder = args.y_save_folder
num_files = args.num_files

# load the split json
split_json = "./data/MR2CT/split.json"

# construct the file list for train/val/test
num_train = int(num_files * train_ratio)
num_val = int(num_files * val_ratio)
num_test = num_files - num_train - num_val
print("num_train: ", num_train, "num_val: ", num_val, "num_test: ", num_test)

x_train_file_list = glob.glob(x_path + "/*.nii.gz")