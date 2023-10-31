# we will receive the total numbers and the ratio of train/val/test

import os
import numpy as np
import nibabel as nib
import argparse
import glob
import tqdm
import json

from nibabel.processing import smooth_image

from util import prepare_folder

#now merge the code into a function

yita_1, yita_2, yita_3, r = 5e-2, 1e-7, 3e-2, 300

def simulate_pvc(img, yita_1, yita_2, yita_3, r):

    data = img.get_fdata()
    data = np.clip(data, 0, 3000)
    data = data / 3000 
    mu_map = data * yita_1 * r
    poisson_img = np.random.poisson(mu_map) / 50
    poisson_noise = np.random.poisson(yita_2 * r, data.shape)
    noisy_img = poisson_img + poisson_noise
    fwhm = yita_3 * r
    pvc_img = smooth_image(nib.Nifti1Image(noisy_img, img.affine, img.header), fwhm=fwhm).get_fdata()

    return pvc_img

def generate_fusion_img(ori_img, pvc_img, t, max_t=1000):
    shifted_t = (t - 1) / max_t
    shifted_t = shifted_t ** (1/3)
    fusion_img = ori_img * (1-shifted_t) + pvc_img * shifted_t
    return fusion_img

parser = argparse.ArgumentParser()
parser.add_argument('--x_path', default='./data/MR2CT/MR', type=str)
parser.add_argument('--train_ratio', default=0.7, type=float)
parser.add_argument('--val_ratio', default=0.2, type=float)
parser.add_argument('--test_ratio', default=0.1, type=float)
parser.add_argument('--seed', default=1024, type=int)
parser.add_argument('--x_save_folder', default='./data/PVC/oriMR_x_2d', type=str)
parser.add_argument('--y_save_folder', default='./data/PVC/pseMR_y_2d', type=str)
parser.add_argument('--num_files', default=100, type=int)
args = parser.parse_args()

# load the parameters
x_path = args.x_path
train_ratio = args.train_ratio
val_ratio = args.val_ratio
test_ratio = args.test_ratio
seed = args.seed
x_save_folder = args.x_save_folder
y_save_folder = args.y_save_folder
num_files = args.num_files

# load the split json
split_json = "./data/PVC/split.json"
with open(split_json, "r") as f:
    datasets = json.load(f)
    train_files = datasets["train"]
    val_files = datasets["val"]
    test_files = datasets["test"]

if num_files > 0:
    # construct the file list for train/val/test
    num_train = int(num_files * train_ratio)
    num_val = int(num_files * val_ratio)
    num_test = num_files - num_train - num_val
    print("num_train: ", num_train, "num_val: ", num_val, "num_test: ", num_test)

    # create the file list for train/val/test
    train_file_list = train_files[0:num_train]
    val_file_list = val_files[0:num_val]
    test_file_list = test_files[0:num_test]
else:
    train_file_list = train_files
    val_file_list = val_files
    test_file_list = test_files

# create the folders for train/val/test
x_save_folder_train = x_save_folder + "/train/"
y_save_folder_train = y_save_folder + "/train/"
x_save_folder_val = x_save_folder + "/val/"
y_save_folder_val = y_save_folder + "/val/"
x_save_folder_test = x_save_folder + "/test/"
y_save_folder_test = y_save_folder + "/test/"
for folder in [x_save_folder_train, y_save_folder_train, x_save_folder_val, y_save_folder_val, x_save_folder_test, y_save_folder_test]:
    os.makedirs(folder, exist_ok=True)

# copy files from the original folder to the new folder
for filepath in train_file_list:
    x_filepath = filepath
    x_img = nib.load(x_filepath)
    new_filename_x = x_save_folder_train + x_filepath.split("/")[-1]
    os.system("cp {} {}".format(x_filepath, new_filename_x))
    print("copy {} to {}".format(x_filepath, new_filename_x))

    x_pvc = simulate_pvc(x_img, yita_1, yita_2, yita_3, r)
    new_filename_y = y_save_folder_train + x_filepath.split("/")[-1]
    # save the pvc image
    nib.save(nib.Nifti1Image(x_pvc, x_img.affine, x_img.header), new_filename_y)
    print("save {} to {}".format(x_filepath, new_filename_y))

for filepath in val_file_list:
    x_filepath = filepath
    x_img = nib.load(x_filepath)
    new_filename_x = x_save_folder_val + x_filepath.split("/")[-1]
    os.system("cp {} {}".format(x_filepath, new_filename_x))
    print("copy {} to {}".format(x_filepath, new_filename_x))

    x_pvc = simulate_pvc(x_img, yita_1, yita_2, yita_3, r)
    new_filename_y = y_save_folder_val + x_filepath.split("/")[-1]
    # save the pvc image
    nib.save(nib.Nifti1Image(x_pvc, x_img.affine, x_img.header), new_filename_y)
    print("save {} to {}".format(x_filepath, new_filename_y))

for filepath in test_file_list:
    x_filepath = filepath
    x_img = nib.load(x_filepath)
    new_filename_x = x_save_folder_test + x_filepath.split("/")[-1]
    os.system("cp {} {}".format(x_filepath, new_filename_x))
    print("copy {} to {}".format(x_filepath, new_filename_x))

    x_pvc = simulate_pvc(x_img, yita_1, yita_2, yita_3, r)
    new_filename_y = y_save_folder_test + x_filepath.split("/")[-1]
    # save the pvc image
    nib.save(nib.Nifti1Image(x_pvc, x_img.affine, x_img.header), new_filename_y)
    print("save {} to {}".format(x_filepath, new_filename_y))

# prepare the parameters for the prepare_folder function
oriMR_folder_list = [
    x_save_folder_train,
    x_save_folder_val,
    x_save_folder_test,
]

pseMR_folder_list = [
    y_save_folder_train,
    y_save_folder_val,
    y_save_folder_test,
]

for data_folder in oriMR_folder_list:
    prepare_folder(data_folder, modality="MR", isDelete=False)

for data_folder in pseMR_folder_list:
    prepare_folder(data_folder, modality="Normed", isDelete=False)