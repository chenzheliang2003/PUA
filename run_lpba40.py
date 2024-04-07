import os

command1 = "CUDA_VISIBLE_DEVICES=0 python3 train.py --dataset LPBA40 --save_frequency 200 --alpha 4 --learning_rate 4e-4 --cascade 2 --tag cascade_2_ape"
command2 = "CUDA_VISIBLE_DEVICES=1 python3 train.py --dataset LPBA40 --save_frequency 200 --alpha 4 --learning_rate 4e-4 --cascade 2 --tag cascade_2_embed_48_deep"
os.system(command1)