# this file is used to preprocess the dataset. We need to do the following things:
# 1. load two folders, one for the input data, one for the output data
# 2. construct the file list for both folders, requiring the file name of the input data and the output data are the same
# 3. construct the train/val/test split according to the given ratio, and save the split as a json file
# 4. create the folders for train/val/test split, and move files into them
# 5. for each folder, run the function prepare_folder to save the files as *.npy 

import os
import json
import numpy as np
import nibabel as nib
import argparse
import glob
import tqdm

def prepare_folder(folder_path, modality="MR"):

# for each folder, do the following process
# here we will load every file with *.nii.gz in the folder given by the user
# and then we will save the file as *.npy, with the following steps:
# 1. load the file
# 2. normalize the file by caping the value to 0-3000, and then divide by 3000
# 3. known the shape of the file is w * h * d, and we save 3 adjacent slices as a group, in the shape of w * h * 3
# 4. save each group as a *.npy file, with the name of the file as the same as the original file plus the slice number
# 5. save the file in the new folder given by the user
# 6. print out how many slices are saved
# 7. repeat the above steps for all the files in the folder

    file_list = glob.glob(folder_path + "*.nii.gz")
    print("file_list: ", file_list)

    # create a new folder to save results, with the given name + "2d"
    save_path = folder_path + "2d/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # create a list to store how many slices in each file
    num_slices = []

    for filename in tqdm.tqdm(file_list):

        # load the file
        img = nib.load(filename)
        img_data = img.get_fdata()

        # if MR, normalize the file by caping the value to 0-3000, and then divide by 3000
        # if CT, normalize the file by caping the value to -1024-3000, and then divide by 4024
        if modality == "MR":
            img_data = np.clip(img_data, 0, 3000)
            img_data = img_data / 3000
        elif modality == "CT":
            img_data = np.clip(img_data, -1024, 3000)
            img_data = img_data / 4024
        else:
            raise ValueError("modality should be MR or CT")

        # known the shape of the file is w * h * d, and we save 3 adjacent slices as a group, in the shape of w * h * 3
        w, h, d = img_data.shape
        img_data_new = np.zeros((w, h, 3))

        # iterate through the slices
        for idx in range(d):

            if idx == 0:
                img_data_new = img_data[:, :, 0:3]
            elif idx == d - 1:
                img_data_new = img_data[:, :, d - 3:d]
            else:
                img_data_new = img_data[:, :, idx - 1:idx + 2]

        

            # save each group as a *.npy file, with the name of the file as the same as the original file plus the slice number
            savename = save_path + filename.split("/")[-1].split(".")[0] + "_{:04d}".format(idx) + ".npy"
            
            # save the file with the formative name of 4 digits in the new folder given by the user
            np.save(savename, img_data_new)

        # save the filename and the number of slices in the file
        num_slices.append([filename, d])

    # print out how many slices are saved
    print("num_slices: ", num_slices)

# load the two folders, and the split ratio and random seed
parser = argparse.ArgumentParser()
parser.add_argument('--x_path', default='./data/MR2CT/MR', type=str)
parser.add_argument('--y_path', default='./data/MR2CT/CT', type=str)
parser.add_argument('--train_ratio', default=0.7, type=float)
parser.add_argument('--val_ratio', default=0.2, type=float)
parser.add_argument('--test_ratio', default=0.1, type=float)
parser.add_argument('--seed', default=1024, type=int)
args = parser.parse_args()
print(args)

# load the two folders, and the split ratio
x_path = args.data_path
y_path = args.save_path
train_ratio = args.train_ratio
val_ratio = args.val_ratio
test_ratio = args.test_ratio

# construct the file list
x_file_list = glob.glob(x_path + "*.nii.gz")
y_file_list = glob.glob(y_path + "*.nii.gz")

# sort the file list
x_file_list.sort()
y_file_list.sort()

# check if the file names are the same
for idx in range(len(x_file_list)):
    x_filename = x_file_list[idx].split("/")[-1].split(".")[0]
    y_filename = y_file_list[idx].split("/")[-1].split(".")[0]
    if x_filename != y_filename:
        raise ValueError("the file names are not the same")
    
# construct the train/val/test split
num_files = len(x_file_list)
num_train = int(num_files * train_ratio)
num_val = int(num_files * val_ratio)
num_test = num_files - num_train - num_val

# create the folders for train/val/test split
train_folder = "./data/MR2CT/train/"
val_folder = "./data/MR2CT/val/"
test_folder = "./data/MR2CT/test/"

train_folder_list = []
val_folder_list = []
test_folder_list = []

# for each split, create MR and CT folders
for folder in [train_folder, val_folder, test_folder]:
    if not os.path.exists(folder):
        os.mkdir(folder)
    if not os.path.exists(folder + "MR/"):
        os.mkdir(folder + "MR/")
    if not os.path.exists(folder + "CT/"):
        os.mkdir(folder + "CT/")

# set the random seed, create a file list and shuffle it
np.random.seed(args.seed)
file_list = []
for idx in range(num_files):
    file_list.append([x_file_list[idx], y_file_list[idx]])
file_list = np.array(file_list)
np.random.shuffle(file_list)

# move files into the folders
for idx in range(num_train):
    x_filename = file_list[idx, 0]
    y_filename = file_list[idx, 1]
    train_folder_list.append([x_filename, y_filename])
    os.system("mv {} {}".format(x_filename, train_folder + "MR/"))
    os.system("mv {} {}".format(y_filename, train_folder + "CT/"))

for idx in range(num_train, num_train + num_val):
    x_filename = file_list[idx, 0]
    y_filename = file_list[idx, 1]
    val_folder_list.append([x_filename, y_filename])
    os.system("mv {} {}".format(x_filename, val_folder + "MR/"))
    os.system("mv {} {}".format(y_filename, val_folder + "CT/"))

for idx in range(num_train + num_val, num_files):
    x_filename = file_list[idx, 0]
    y_filename = file_list[idx, 1]
    test_folder_list.append([x_filename, y_filename])
    os.system("mv {} {}".format(x_filename, test_folder + "MR/"))
    os.system("mv {} {}".format(y_filename, test_folder + "CT/"))

# for each folder, run the function prepare_folder to save the files as *.npy
for folder in [train_folder, val_folder, test_folder]:
    prepare_folder(folder + "MR/", modality="MR")
    prepare_folder(folder + "CT/", modality="CT")

# save the split as a json file, and save the json file
json_file = "./data/MR2CT/split.json"
split_dict = {
    "train": train_folder_list,
    "val": val_folder_list,
    "test": test_folder_list
}
with open(json_file, "w") as f:
    json.dump(split_dict, f, indent=4)
f.close()








