import os

command = "CUDA_VISIBLE_DEVICES=1 python3 train.py --dataset IXI --batch_size 1 --num_epochs 500 --save_frequency 200 --alpha 4 --learning_rate 4e-4 --cascade 2"
os.system(command)