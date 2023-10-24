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
with open(split_json, "r") as f:
    datasets = json.load(f)
    train_files = datasets["train"]
    val_files = datasets["validation"]
    test_files = datasets["test"]

# construct the file list for train/val/test
num_train = int(num_files * train_ratio)
num_val = int(num_files * val_ratio)
num_test = num_files - num_train - num_val
print("num_train: ", num_train, "num_val: ", num_val, "num_test: ", num_test)

# create the file list for train/val/test
train_file_list = train_files[0:num_train]
val_file_list = val_files[0:num_val]
test_file_list = test_files[0:num_test]

# create the folders for train/val/test
x_save_folder_train = x_save_folder + "/train/"
y_save_folder_train = y_save_folder + "/train/"
x_save_folder_val = x_save_folder + "/val/"
y_save_folder_val = y_save_folder + "/val/"
x_save_folder_test = x_save_folder + "/test/"
y_save_folder_test = y_save_folder + "/test/"
for folder in [x_save_folder_train, y_save_folder_train, x_save_folder_val, y_save_folder_val, x_save_folder_test, y_save_folder_test]:
    if not os.path.exists(folder):
        os.mkdir(folder)
        print("create folder: ", folder)

# copy files from the original folder to the new folder
for filepath_set in train_file_list:
    x_filepath = filepath_set["MR"]
    y_filepath = filepath_set["CT"]
    new_filename_x = x_save_folder_train + x_filepath.split("/")[-1]
    new_filename_y = y_save_folder_train + y_filepath.split("/")[-1]
    os.system("cp {} {}".format(x_filepath, new_filename_x))
    os.system("cp {} {}".format(y_filepath, new_filename_y))
    print("copy {} to {}".format(x_filepath, new_filename_x))
    print("copy {} to {}".format(y_filepath, new_filename_y))

for filepath_set in val_file_list:
    x_filepath = filepath_set["MR"]
    y_filepath = filepath_set["CT"]
    new_filename_x = x_save_folder_val + x_filepath.split("/")[-1]
    new_filename_y = y_save_folder_val + y_filepath.split("/")[-1]
    os.system("cp {} {}".format(x_filepath, new_filename_x))
    os.system("cp {} {}".format(y_filepath, new_filename_y))
    print("copy {} to {}".format(x_filepath, new_filename_x))
    print("copy {} to {}".format(y_filepath, new_filename_y))

for filepath_set in test_file_list:
    x_filepath = filepath_set["MR"]
    y_filepath = filepath_set["CT"]
    new_filename_x = x_save_folder_test + x_filepath.split("/")[-1]
    new_filename_y = y_save_folder_test + y_filepath.split("/")[-1]
    os.system("cp {} {}".format(x_filepath, new_filename_x))
    os.system("cp {} {}".format(y_filepath, new_filename_y))
    print("copy {} to {}".format(x_filepath, new_filename_x))
    print("copy {} to {}".format(y_filepath, new_filename_y))

# prepare the parameters for the prepare_folder function
data_folder_list = [
    x_save_folder_train,
    x_save_folder_val,
    x_save_folder_test,
    y_save_folder_train,
    y_save_folder_val,
    y_save_folder_test,
]

for data_folder in data_folder_list:
    prepare_folder(data_folder, modality="MR", isDelete=True)