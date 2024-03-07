# MR data foder:
# data/MR2CT/train/MR
# data/MR2CT/val/MR
# data/MR2CT/test/MR
# data/MR2CT/test/MR/old_train

# CT data foder:
# data/MR2CT/train/CT
# data/MR2CT/val/CT
# data/MR2CT/test/CT
# data/MR2CT/test/CT/old_train

# Target folder:
# data/MR2CT/WIMR

# There are .nii.gz in both folder with the same name
# copy each data to target folder in separate folder
# for example, there are /MR_folder/00001.nii.gz and /CT_folder/00001.nii.gz
# we will copy /target_folder/00001/mr.nii.gz and /target_folder/00001/ct.nii.gz

import os
import glob

mr_data_folder_list = [
    "data/MR2CT/train/MR",
    "data/MR2CT/val/MR",
    "data/MR2CT/test/MR",
    "data/MR2CT/test/MR/old_train",
]

ct_data_folder_list = [
    "data/MR2CT/train/CT",
    "data/MR2CT/val/CT",
    "data/MR2CT/test/CT",
    "data/MR2CT/test/CT/old_train",
]

target_folder = "data/MR2CT/WIMR"

for mr_data_folder, ct_data_folder in zip(mr_data_folder_list, ct_data_folder_list):
    mr_list = sorted(glob.glob(os.path.join(mr_data_folder, "*.nii.gz")))
    ct_list = sorted(glob.glob(os.path.join(ct_data_folder, "*.nii.gz")))
    print(f"Find {len(mr_list)} MR data and {len(ct_list)} CT data in {mr_data_folder} and {ct_data_folder}.")

    for mr, ct in zip(mr_list, ct_list):
        mr_name = os.path.basename(mr).split(".")[0]
        ct_name = os.path.basename(ct).split(".")[0]
        # check if the name is the same
        if mr_name != ct_name:
            print(f"MR name {mr_name} is not the same as CT name {ct_name}.")
            continue
        target_mr_folder = os.path.join(target_folder, mr_name)
        target_ct_folder = os.path.join(target_folder, ct_name)
        os.makedirs(target_mr_folder, exist_ok=True)
        os.makedirs(target_ct_folder, exist_ok=True)
        os.system(f"cp {mr} {target_mr_folder}/mr.nii.gz")
        os.system(f"cp {ct} {target_ct_folder}/ct.nii.gz")
        print(f"Copy {mr} to {target_mr_folder}/mr.nii.gz")
        print(f"Copy {ct} to {target_ct_folder}/ct.nii.gz")