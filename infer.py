import glob
import os, losses, utils
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from natsort import natsorted
from models.TransMorph import CONFIGS as CONFIGS_TM
import models.TransMorph as TransMorph
import torch.nn as nn

def main():
    atlas_dir = '../../../../IXI_data/atlas.pkl'
    test_dir = '../../../../IXI_data/Test/'
    model_idx = -1
    weights = [1, 1]
    model_folder = 'TransMorph_ncc_{}_diffusion_{}/'.format(weights[0], weights[1])
    model_dir = 'experiments/' + model_folder
    if 'Val' in test_dir:
        csv_name = model_folder[:-1]+'_Val'
    else:
        csv_name = model_folder[:-1]
    dict = utils.process_label()
    if not os.path.exists('Quantitative_Results/'):
        os.makedirs('Quantitative_Results/')
    if os.path.exists('Quantitative_Results/'+csv_name+'.csv'):
        os.remove('Quantitative_Results/'+csv_name+'.csv')
    csv_writter(model_folder[:-1], 'Quantitative_Results/' + csv_name)
    line = ''
    for i in range(46):
        line = line + ',' + dict[i]
    csv_writter(line +','+'non_jec', 'Quantitative_Results/' + csv_name)

    config = CONFIGS_TM['TransMorph']
    model = TransMorph.TransMorph(config)
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])['state_dict']
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)
    model.cuda()
    reg_model = utils.register_model(config.img_size, 'bilinear')
    reg_model.cuda()
    test_composed = transforms.Compose([trans.Seg_norm(),
                                        trans.NumpyType((np.float32, np.int16)),
                                        ])
    test_set = datasets.IXIBrainInferDataset(glob.glob(test_dir + '*.pkl'), atlas_dir, transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    eval_dsc_def = utils.AverageMeter()
    eval_dsc_raw = utils.AverageMeter()
    eval_det = utils.AverageMeter()
    with torch.no_grad():
        stdy_idx = 0
        for data in test_loader:
            model.eval()
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]

            x_in = torch.cat((x,y),dim=1)
            x_def, flow = model(x_in)
            x_seg_oh = nn.functional.one_hot(x_seg.long(), num_classes=46)
            x_seg_oh = torch.squeeze(x_seg_oh, 1)
            x_seg_oh = x_seg_oh.permute(0, 4, 1, 2, 3).contiguous()
            #x_segs = model.spatial_trans(x_seg.float(), flow.float())
            x_segs = []
            for i in range(46):
                def_seg = reg_model([x_seg_oh[:, i:i + 1, ...].float(), flow.float()])
                x_segs.append(def_seg)
            x_segs = torch.cat(x_segs, dim=1)
            def_out = torch.argmax(x_segs, dim=1, keepdim=True)
            del x_segs, x_seg_oh
            #def_out = reg_model([x_seg.cuda().float(), flow.cuda()])
            tar = y.detach().cpu().numpy()[0, 0, :, :, :]
            jac_det = utils.jacobian_determinant_vxm(flow.detach().cpu().numpy()[0, :, :, :, :])
            line = utils.dice_val_substruct(def_out.long(), y_seg.long(), stdy_idx)
            line = line +','+str(np.sum(jac_det <= 0)/np.prod(tar.shape))
            csv_writter(line, 'Quantitative_Results/' + csv_name)
            eval_det.update(np.sum(jac_det <= 0) / np.prod(tar.shape), x.size(0))
            print('det < 0: {}'.format(np.sum(jac_det <= 0) / np.prod(tar.shape)))
            dsc_trans = utils.dice_val(def_out.long(), y_seg.long(), 46)
            dsc_raw = utils.dice_val(x_seg.long(), y_seg.long(), 46)
            print('Trans dsc: {:.4f}, Raw dsc: {:.4f}'.format(dsc_trans.item(),dsc_raw.item()))
            eval_dsc_def.update(dsc_trans.item(), x.size(0))
            eval_dsc_raw.update(dsc_raw.item(), x.size(0))
            stdy_idx += 1

        print('Deformed DSC: {:.3f} +- {:.3f}, Affine DSC: {:.3f} +- {:.3f}'.format(eval_dsc_def.avg,
                                                                                    eval_dsc_def.std,
                                                                                    eval_dsc_raw.avg,
                                                                                    eval_dsc_raw.std))
        print('deformed det: {}, std: {}'.format(eval_det.avg, eval_det.std))

def csv_writter(line, name):
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')

if __name__ == '__main__':
    main()