import os
import sys
import glob
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models import PUA, SpatialTransformer
import losses
import utils

def main():
    args = utils.parse_arguments()
    if args.tag is None:
        save_dir = "PUA_{}_grad_{}/".format(args.dataset, args.alpha)
    else:
        save_dir = "PUA_{}_grad_{}_tag_{}/".format(args.dataset, args.alpha, args.tag)
    if not os.path.exists("experiments/"+save_dir):
        os.makedirs("experiments/"+save_dir)
    if not os.path.exists("logs/"+save_dir):
        os.makedirs("logs/"+save_dir)
    sys.stdout = utils.Logger("logs/"+save_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dataset == "LPBA40":
        train_path = "../LPBA40_data/train/"
        train_label_path = "../LPBA40_data/train_label/"
        test_path = "../LPBA40_data/test/"
        test_label_path = "../LPBA40_data/test_label/"
        atlas_path = "../LPBA40_data/atlas.nii.gz"
        atlas_label_path = "../LPBA40_data/atlas_label.nii.gz"
        img_size = (160, 192, 160)
        window_size = (5, 6, 5)
        train_set = utils.LPBA40Dataset(glob.glob(train_path + "*"), atlas_path, glob.glob(train_label_path + "*"), atlas_label_path)
        val_set = utils.LPBA40Dataset(glob.glob(test_path + "*"), atlas_path, glob.glob(test_label_path + "*"), atlas_label_path)
    else:
        train_path = "../IXI_data/Train/"
        val_path = "../IXI_data/Val/"
        atlas_path = "../IXI_data/atlas.pkl"
        img_size = (160, 192, 224)
        window_size = (5, 6, 7)
        train_set = utils.IXIDataset(glob.glob(train_path + "*"), atlas_path)
        val_set = utils.IXIDataset(glob.glob(val_path + "*"), atlas_path)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    model = PUA(img_size=img_size, window_size=window_size)
    if device == "cuda":
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count())).to("cuda")
    warp = SpatialTransformer(img_size=img_size)
    warp_label = SpatialTransformer(img_size=img_size, mode="nearest")
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)
    writer = SummaryWriter(log_dir="logs/"+save_dir)
    for epoch in range(args.num_epochs):
        loss_all = utils.Accumulater()
        idx = 0
        for data in train_loader:
            idx += 1
            model.train()
            warp.train()
            moving, fixed, _, _ = [t.to(device) for t in data]
            grad_loss = 0
            for cascade in range(args.cascade):
                flow_field = model(moving, fixed)
                grad_loss += losses.grad_loss(flow_field) * args.alpha
                moving = warp(moving, flow_field)
            sim_loss = losses.ncc_loss(moving, fixed)
            loss = sim_loss + grad_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=.5)
            optimizer.step()
            loss_all.update(loss.item(), args.batch_size)
            print("Epoch {} - Iter {} of {} Loss {:.6f}, Sim Loss: {:.6f}, Grad Loss: {:.6f}".format(
                epoch + 1, idx, len(train_loader), loss.item(), sim_loss.item(), grad_loss.item()))
        writer.add_scalar("Loss/train", loss_all.avg, epoch)
        print("Epoch {} Loss {:.6f}".format(epoch + 1, loss_all.avg))
        eval_dsc = utils.Accumulater()
        idx = 0
        with torch.no_grad():
            for data in val_loader:
                idx += 1
                model.eval()
                warp_label.eval()
                moving, fixed, moving_label, fixed_label = [t.to(device) for t in data]
                for cascade in range(args.cascade):
                    flow_field = model(moving, fixed)
                    moving = warp(moving, flow_field)
                    moving_label = warp_label(moving_label, flow_field)
                dsc = utils.get_label_dice(moving_label, fixed_label, dataset=args.dataset)
                eval_dsc.update(dsc.item(), args.batch_size)
                print("Epoch {} - Iter {} of {} Dsc: {:.6f}".format(
                    epoch + 1, idx, len(val_loader), dsc.item()))
        print("Epoch {} - Dsc Avg: {:.6f}, Std: {:.6f}".format(epoch + 1, eval_dsc.avg, eval_dsc.std))
        if epoch % args.save_frequency == 0:
            utils.save_checkpoint({
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "dsc": eval_dsc.avg,
                "optimizer": optimizer.state_dict()
            }, save_dir="experiments/" + save_dir, filename="dsc{:.3f}.pth.tar".format(eval_dsc.avg))
        writer.add_scalar("DSC/validate", eval_dsc.avg, epoch)
        loss_all.reset()
        eval_dsc.reset()
    writer.close()

if __name__ == "__main__":
    main()