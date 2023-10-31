# this file is used to preprocess the dataset. We need to do the following things:
# 1. load two folders, one for the input data, one for the output data
# 2. construct the file list for both folders, requiring the file name of the input data and the output data are the same
# 3. construct the train/val/test split according to the given ratio, and save the split as a json file
# 4. create the folders for train/val/test split, and move files into them
# 5. for each folder, run the function prepare_folder to save the files as *.npy 

import os
import json
import numpy as np
import argparse
import glob

# from util import prepare_folder

# load the two folders, and the split ratio and random seed
parser = argparse.ArgumentParser()
parser.add_argument('--x_path', default='./data/MR2CT/MR', type=str)
parser.add_argument('--train_ratio', default=0.7, type=float)
parser.add_argument('--val_ratio', default=0.2, type=float)
parser.add_argument('--test_ratio', default=0.1, type=float)
parser.add_argument('--seed', default=1024, type=int)
args = parser.parse_args()
print(args)

# load the two folders, and the split ratio
x_path = args.x_path
train_ratio = args.train_ratio
val_ratio = args.val_ratio
test_ratio = args.test_ratio

# construct the file list
x_file_list = glob.glob(x_path + "/*.nii.gz")
print("MR files num: ", len(x_file_list))

# sort the file list
x_file_list.sort()
    
# construct the train/val/test split
num_files = len(x_file_list)
num_train = int(num_files * train_ratio)
num_val = int(num_files * val_ratio)
num_test = num_files - num_train - num_val
print("num_train: ", num_train, "num_val: ", num_val, "num_test: ", num_test)

# create the folders for train/val/test split
train_folder = "./data/PVC/train/"
val_folder = "./data/PVC/val/"
test_folder = "./data/PVC/test/"

train_folder_list = []
val_folder_list = []
test_folder_list = []

# for each split, create MR and CT folders
for folder in [train_folder, val_folder, test_folder]:
    if not os.path.exists(folder):
        os.mkdir(folder)
        print("create folder: ", folder)

# set the random seed, create a file list and shuffle it
np.random.seed(args.seed)
file_list = []
for idx in range(num_files):
    file_list.append(x_file_list[idx])
file_list = np.array(file_list)
np.random.shuffle(file_list)

# move files into the folders
for idx in range(num_train):
    x_filename = file_list[idx]
    savename_x = train_folder + "/" + os.path.basename(x_filename)
    train_folder_list.append({
        "MR": savename_x,
    })
    os.system("mv {} {}".format(x_filename, savename_x))
    print("mv {} {}".format(x_filename, savename_x))

for idx in range(num_train, num_train + num_val):
    x_filename = file_list[idx]
    savename_x = val_folder + "/" + os.path.basename(x_filename)
    val_folder_list.append({
        "MR": savename_x,
    })
    os.system("mv {} {}".format(x_filename, savename_x))
    print("mv {} {}".format(x_filename, savename_x))

for idx in range(num_train + num_val, num_files):
    x_filename = file_list[idx]
    savename_x = test_folder + "/" + os.path.basename(x_filename)
    test_folder_list.append({
        "MR": savename_x,
    })
    os.system("mv {} {}".format(x_filename, savename_x))
    print("mv {} {}".format(x_filename, savename_x))

# save the split as a json file, and save the json file
json_file = "./data/PVC/split.json"
split_dict = {
    "train": train_folder_list,
    "val": val_folder_list,
    "test": test_folder_list
}
with open(json_file, "w") as f:
    json.dump(split_dict, f, indent=4)
f.close()
print("save json file: ", json_file)

# # for each folder, run the function prepare_folder to save the files as *.npy
# for folder in [train_folder, val_folder, test_folder]:
#     prepare_folder(folder + "MR/", modality="MR")
#     prepare_folder(folder + "CT/", modality="CT")









