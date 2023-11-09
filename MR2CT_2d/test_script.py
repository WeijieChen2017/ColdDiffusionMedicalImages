import os

cmd_list = [
    "python MR2CT_simpleUNet_test.py --load_path ./proj/MR2CT_typeA/model_200000.pt --n_jumps 1"
    "python MR2CT_simpleUNet_test.py --load_path ./proj/MR2CT_typeA/model_200000.pt --n_jumps 2"
    "python MR2CT_simpleUNet_test.py --load_path ./proj/MR2CT_typeA/model_200000.pt --n_jumps 3"
    "python MR2CT_simpleUNet_test.py --load_path ./proj/MR2CT_typeA/model_200000.pt --n_jumps 4"
    "python MR2CT_simpleUNet_test.py --load_path ./proj/MR2CT_typeA/model_200000.pt --n_jumps 5"
    "python MR2CT_simpleUNet_test.py --load_path ./proj/MR2CT_typeA/model_200000.pt --n_jumps 9"
    "python MR2CT_simpleUNet_test.py --load_path ./proj/MR2CT_typeB/model_200000.pt --n_jumps 1"
    "python MR2CT_simpleUNet_test.py --load_path ./proj/MR2CT_typeB/model_200000.pt --n_jumps 2"
    "python MR2CT_simpleUNet_test.py --load_path ./proj/MR2CT_typeB/model_200000.pt --n_jumps 3"
    "python MR2CT_simpleUNet_test.py --load_path ./proj/MR2CT_typeB/model_200000.pt --n_jumps 4"
    "python MR2CT_simpleUNet_test.py --load_path ./proj/MR2CT_typeB/model_200000.pt --n_jumps 5"
    "python MR2CT_simpleUNet_test.py --load_path ./proj/MR2CT_typeB/model_200000.pt --n_jumps 9"
]

for cmd in cmd_list:
    print(cmd)
    os.system(cmd)