import os
import glob
import tqdm
import nibabel as nib
import numpy as np

def prepare_folder(data_folder, modality="MR", isDelete=False):

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

    file_list = glob.glob(data_folder + "*.nii.gz")
    print("file_list: ", file_list)

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
            savename = os.join(data_folder, filename.split("/")[-1].split(".")[0] + "_{:04d}".format(idx) + ".npy")
            
            # save the file with the formative name of 4 digits in the new folder given by the user
            np.save(savename, img_data_new)

        # save the filename and the number of slices in the file
        num_slices.append([filename, d])

        if isDelete:
            os.remove(filename)
            print("delete: ", filename)

    # print out how many slices are saved
    print("num_slices: ", num_slices)