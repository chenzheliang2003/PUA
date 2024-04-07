import os
import sys
import glob
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import pickle
import SimpleITK as sitk
from natsort import natsorted
import matplotlib.pyplot as plt


class LPBA40Dataset(Dataset):
    
    def __init__(self, data_path, atlas_path, data_label_path, atlas_label_path):
        self.data_path = data_path
        self.data_label_path = data_label_path
        self.atlas_path = atlas_path
        self.atlas_label_path = atlas_label_path

    def __getitem__(self, index):
        moving = sitk.GetArrayFromImage(sitk.ReadImage(self.data_path[index]))
        fixed = sitk.GetArrayFromImage(sitk.ReadImage(self.atlas_path))
        moving_label = sitk.GetArrayFromImage(sitk.ReadImage(self.data_label_path[index]))
        fixed_label = sitk.GetArrayFromImage(sitk.ReadImage(self.atlas_label_path))
        moving = torch.from_numpy(moving).unsqueeze(0).float()
        fixed = torch.from_numpy(fixed).unsqueeze(0).float()
        moving_label = torch.from_numpy(moving_label).unsqueeze(0).float()
        fixed_label = torch.from_numpy(fixed_label).unsqueeze(0).float()
        return moving, fixed, moving_label, fixed_label

    def __len__(self):
        return len(self.data_path)


class IXIDataset(Dataset):

    def __init__(self, data_path, atlas_path):
        self.data_path = data_path
        self.atlas_path = atlas_path

    def __getitem__(self, index):
        with open(self.atlas_path, "rb") as f:
            moving, moving_label = pickle.load(f)
        with open(self.data_path[index], "rb") as f:
            fixed, fixed_label = pickle.load(f)
        self.seg_table = np.array([0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26,
                          28, 30, 31, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 62,
                          63, 72, 77, 80, 85, 251, 252, 253, 254, 255])
        moving_label_processed = np.zeros_like(moving_label)
        for i in range(len(self.seg_table)):
            moving_label_processed[moving_label == self.seg_table[i]] = i
        fixed_label_processed = np.zeros_like(fixed_label)
        for i in range(len(self.seg_table)):
            fixed_label_processed[fixed_label == self.seg_table[i]] = i
        moving = torch.from_numpy(moving).unsqueeze(0).float()
        fixed = torch.from_numpy(fixed).unsqueeze(0).float()
        moving_label_processed = torch.from_numpy(moving_label_processed).unsqueeze(0).float()
        fixed_label_processed = torch.from_numpy(fixed_label_processed).unsqueeze(0).float()
        return moving, fixed, moving_label_processed, fixed_label_processed

    def __len__(self):
        return len(self.data_path)


class Accumulater(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.vals = []
        self.sum = 0
        self.avg = 0
        self.std = 0

    def update(self, val, cnt):
        self.count += cnt
        for _ in range(cnt):
            self.vals.append(val)
        self.sum += val * cnt
        self.avg = self.sum / self.count
        self.std = np.std(self.vals)


class Logger(object):

    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir+"logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        pass


def parse_arguments():
    parser = argparse.ArgumentParser(description="Training Hyperparameters")
    
    parser.add_argument("--dataset", type=str, default="LPBA40",
                        help="Name of the dataset")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=1000,
                        help="Number of training epochs")
    parser.add_argument("--save_frequency", type=int, default=50,
                        help="Frequency of saving checkpoints")
    parser.add_argument("--alpha", type=float, default=1,
                        help="Regularization ratio")
    parser.add_argument("--learning_rate", type=float, default=4e-4,
                        help="Learning rate")
    parser.add_argument("--tag", type=str, default=None,
                        help="Tag for saving dir")
    parser.add_argument("--cascade", type=int, default=2,
                        help="Cascade for warping")
    args = parser.parse_args()
    return args


def dice_score(x, y):
    x = x.flatten()
    y = y.flatten()
    return (2. * (x * y).sum() + 1e-5) / (x.sum() + y.sum() + 1e-5)


def get_label_dice(x, y, dataset="LPBA40"):
    if dataset == "LPBA40":
        cls_lst = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 61, 62,
                63, 64, 65, 66, 67, 68, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 101, 102, 121, 122, 161, 162,
                163, 164, 165, 166]
    else:
        cls_lst = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 34, 36]
    dice_lst = []
    for cls in cls_lst:
        dice_lst.append(dice_score(x == cls, y == cls).item())
    return np.mean(dice_lst)


def save_checkpoint(state, save_dir="models", filename="checkpoint.pth.tar", max_model_num=8):
    torch.save(state, save_dir+filename)
    model_lists = natsorted(glob.glob(save_dir + "*"))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + "*"))


def visualize(x, path):
    x = x.squeeze()
    slices = 1
    for i in range(0, x.shape[0], 10):
        plt.subplot(4, 5, slices)
        plt.imshow(x[:, :, i].squeeze().cpu().detach(), cmap="gray")
        plt.axis("off")
        slices += 1
    f = plt.gcf()
    f.savefig(path)