import os

# command = "CUDA_VISIBLE_DEVICES=0 python3 train.py --dataset LPBA40 --save_frequency 200 --alpha 4 --learning_rate 4e-4 --cascade 2 --tag 2"
command = "CUDA_VISIBLE_DEVICES=0 python3 infer.py --dataset LPBA40 --save_frequency 200 --alpha 4 --learning_rate 4e-4 --cascade 2"
os.system(command)