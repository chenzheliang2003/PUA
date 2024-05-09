import os
import sys
import glob
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from natsort import natsorted
from models import PUA, SpatialTransformer
import utils

def main():
    args = utils.parse_arguments()
    if not os.path.exists("data/"):
        os.makedirs("data/")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dataset == "LPBA40":
        test_path = "../LPBA40_data/test/"
        test_label_path = "../LPBA40_data/test_label/"
        atlas_path = "../LPBA40_data/atlas.nii.gz"
        atlas_label_path = "../LPBA40_data/atlas_label.nii.gz"
        img_size = (160, 192, 160)
        window_size = (5, 6, 5)
        test_set = utils.LPBA40Dataset(glob.glob(test_path + "*"), atlas_path, glob.glob(test_label_path + "*"), atlas_label_path)
        label_info = utils.label_info("LPBA40")
    else:
        test_path = "../IXI_data/Test/"
        atlas_path = "../IXI_data/atlas.pkl"
        img_size = (160, 192, 224)
        window_size = (5, 6, 7)
        test_set = utils.IXIDataset(glob.glob(test_path + "*"), atlas_path)
        label_info = utils.label_info("IXI")
    model_dir = "experiments/" + args.model_dir + "/"
    model = PUA(img_size=img_size, window_size=window_size).to(device)
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-1])
    new_state_dict = {}
    for key, value in best_model['state_dict'].items():
        new_key = key.replace("module.", "")
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)
    warp = SpatialTransformer(img_size=img_size).to(device)
    warp_label = SpatialTransformer(img_size=img_size, mode="nearest").to(device)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    eval_dsc = utils.Accumulater()
    eval_nj = utils.Accumulater()
    with torch.no_grad():
        idx = 0
        dsc_table = []
        for data in test_loader:
            model.eval()
            warp.eval()
            warp_label.eval()
            moving, fixed, moving_label, fixed_label = [t.to(device) for t in data]
            nj = 0
            for cascade in range(args.cascade):
                flow_field = model(moving, fixed)
                moving = warp(moving, flow_field)
                moving_label = warp_label(moving_label, flow_field)
                utils.save_nifti(flow_field.permute(0, 2, 3, 4, 1), "data/flow_field_{}_{}.nii.gz".format(idx, cascade))
                nj += utils.negative_jacobian(flow_field)
            utils.save_nifti(moving, "data/registered_{}.nii.gz".format(idx))
            utils.save_nifti(moving_label, "data/registered_label_{}.nii.gz".format(idx))
            dsc_list = utils.label_dice_score(moving_label, fixed_label, dataset=args.dataset, return_list=True)
            for i, dsc in enumerate(dsc_list):
                dsc_table.append([label_info[i], "PUA", dsc])
            eval_dsc.update(np.mean(dsc).item(), args.batch_size)
            eval_nj.update(nj.item(), args.batch_size)
            idx += 1
        print("Dsc Avg: {:.6f}, Std: {:.6f}".format(eval_dsc.avg, eval_dsc.std))
        print("Nj Avg: {:.6f}, Std: {:.6f}".format(eval_nj.avg, eval_nj.std))
        utils.save_csv(dsc_table, "data/PUA_dice_scores.csv")

if __name__ == '__main__':
    main()
