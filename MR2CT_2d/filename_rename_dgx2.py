import os
import re
import shutil

# Replace 'your_directory_path' with the path to the directory containing your .nii.gz files.
your_directory_path = './data/MR2CT/dgx2/'

# Create the 'MR' and 'CT' directories if they don't exist
mr_directory = os.path.join(your_directory_path, 'MR')
ct_directory = os.path.join(your_directory_path, 'CT')

if not os.path.exists(mr_directory):
    os.makedirs(mr_directory)

if not os.path.exists(ct_directory):
    os.makedirs(ct_directory)

# Updated patterns to match the file names
ct_pattern = re.compile(r"MIMRTL-(\d+)__CT_reg_positive.nii.gz")
mr_pattern = re.compile(r"MIMRTL-(\d+)___MR_Resample.nii.gz")

# Walk through the directory and rename the files accordingly
for filename in os.listdir(your_directory_path):
    print(filename)

    # Check if the file is a CT file
    ct_match = ct_pattern.match(filename)
    if ct_match:
        number = ct_match.group(1)
        new_filename = f"{number}.nii.gz"
        new_path = os.path.join(ct_directory, new_filename)
        old_path = os.path.join(your_directory_path, filename)
        shutil.move(old_path, new_path)

    # Check if the file is an MR file
    mr_match = mr_pattern.match(filename)
    if mr_match:
        number = mr_match.group(1)
        new_filename = f"{number}.nii.gz"
        new_path = os.path.join(mr_directory, new_filename)
        old_path = os.path.join(your_directory_path, filename)
        shutil.move(old_path, new_path)

# This script does not produce an output. It will organize your files into 'MR' and 'CT' folders.
