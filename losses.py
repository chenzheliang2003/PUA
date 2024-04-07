import torch
import torch.nn.functional as F

def ncc_loss(x, y):
    filt = torch.ones([1, 1, 9, 9, 9]).to(x.device)
    x2, y2, xy = x * x, y * y, x * y
    x_sum = F.conv3d(x, filt, padding=(4, 4, 4))
    y_sum = F.conv3d(y, filt, padding=(4, 4, 4))
    x2_sum = F.conv3d(x2, filt, padding=(4, 4, 4))
    y2_sum = F.conv3d(y2, filt, padding=(4, 4, 4))
    xy_sum = F.conv3d(xy, filt, padding=(4, 4, 4))
    x_avg = x_sum / 729
    y_avg = y_sum / 729
    cross = xy_sum - y_avg * x_sum - x_avg * y_sum + x_avg * y_avg * 729
    x_var = x2_sum - 2 * x_avg * x_sum + x_avg * x_avg * 729
    y_var = y2_sum - 2 * y_avg * y_sum + y_avg * y_avg * 729
    return -torch.mean(cross * cross / (x_var * y_var + 1e-5))


def grad_loss(flow_field):
    dx = flow_field[..., 1:, :, :] - flow_field[..., :-1, :, :]
    dy = flow_field[..., :, 1:, :] - flow_field[..., :, :-1, :]
    dz = flow_field[..., :, :, 1:] - flow_field[..., :, :, :-1]
    return ((dx ** 2).mean() + (dy ** 2).mean() + (dz ** 2).mean()) / 3
