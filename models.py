import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.distributions.normal import Normal
import numpy as np
from mamba_ssm import Mamba
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    B, H, W, D, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], D // window_size[2],
               window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, *window_size, C)
    return windows


def window_reverse(windows, window_size, H, W, D):
    B = int(windows.shape[0] / (H * W * D / window_size[0] / window_size[1] / window_size[2]))
    x = windows.view(B, H // window_size[0], W // window_size[1], D // window_size[2], *window_size, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, H, W, D, -1)
    return x


class WindowSelfAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)
        self.cpb_mlp = nn.Sequential(
            nn.Linear(3, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False))
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_d = torch.arange(-(self.window_size[2] - 1), self.window_size[2], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h,
                            relative_coords_w,
                            relative_coords_d], indexing="ij")).permute(1, 2, 3, 0).contiguous().unsqueeze(0)
        relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
        relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table[:, :, :, 2] /= (self.window_size[2] - 1)
        relative_coords_table *= 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)
        self.register_buffer("relative_coords_table", relative_coords_table)
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords_d = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w, coords_d], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / .01, device=self.logit_scale.device))).exp()
        attn = attn * logit_scale
        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class WindowCrossAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)
        self.cpb_mlp = nn.Sequential(
            nn.Linear(3, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False))
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_d = torch.arange(-(self.window_size[2] - 1), self.window_size[2], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h,
                            relative_coords_w,
                            relative_coords_d], indexing="ij")).permute(1, 2, 3, 0).contiguous().unsqueeze(0)
        relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
        relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table[:, :, :, 2] /= (self.window_size[2] - 1)
        relative_coords_table *= 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)
        self.register_buffer("relative_coords_table", relative_coords_table)
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords_d = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w, coords_d], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv1 = nn.Linear(dim, dim * 3, bias=False)
        self.qkv2 = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q1_bias = nn.Parameter(torch.zeros(dim))
            self.v1_bias = nn.Parameter(torch.zeros(dim))
            self.q2_bias = nn.Parameter(torch.zeros(dim))
            self.v2_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q1_bias = None
            self.v1_bias = None
            self.q2_bias = None
            self.v2_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y, mask=None):
        B_, N, C = x.shape
        qkv1_bias = None
        qkv2_bias = None
        if self.q1_bias is not None:
            qkv1_bias = torch.cat((self.q1_bias, torch.zeros_like(self.v1_bias, requires_grad=False), self.v1_bias))
            qkv2_bias = torch.cat((self.q2_bias, torch.zeros_like(self.v2_bias, requires_grad=False), self.v2_bias))
        qkv1 = F.linear(input=x, weight=self.qkv1.weight, bias=qkv1_bias)
        qkv1 = qkv1.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        qkv2 = F.linear(input=y, weight=self.qkv2.weight, bias=qkv2_bias)
        qkv2 = qkv2.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv2[0], qkv1[1], qkv1[2]
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / .01, device=self.logit_scale.device))).exp()
        attn = attn * logit_scale
        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        cross = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        cross = self.proj(cross)
        cross = self.proj_drop(cross)
        return cross


class SelfSwinTransformerBlock(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, window_size=(7, 7, 7), shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)
        self.self_attn = WindowSelfAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        if min(self.shift_size) > 0:
            H, W, D = self.input_resolution
            img_mask = torch.zeros((1, H, W, D, 1))
            h_slices = (slice(0, -self.window_size[0]),
                        slice(-self.window_size[0], -self.shift_size[0]),
                        slice(-self.shift_size[0], None))
            w_slices = (slice(0, -self.window_size[1]),
                        slice(-self.window_size[1], -self.shift_size[1]),
                        slice(-self.shift_size[1], None))
            d_slices = (slice(0, -self.window_size[2]),
                        slice(-self.window_size[2], -self.shift_size[2]),
                        slice(-self.shift_size[2], None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    for d in d_slices:
                        img_mask[:, h, w, d, :] = cnt
                        cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2])
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W, D = self.input_resolution
        B, _, C = x.shape
        shortcut = x
        x = x.view(B, H, W, D, C)
        if min(self.shift_size) > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C)
        attn_windows = self.self_attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, *self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W, D)
        if min(self.shift_size) > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]),
                           dims=(1, 2, 3))
        else:
            x = shifted_x
        x = x.view(B, H * W * D, C)
        x = shortcut + self.drop_path(self.norm1(x))
        x = x + self.drop_path(self.norm2(self.mlp(x)))
        return x


class MambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm = norm_layer(dim)

    def forward(self, x):
        return self.norm(self.mamba(x))


class CrossSwinTransformerBlock(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, window_size=(7, 7, 7), shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)
        self.cross_attn = WindowCrossAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        if min(self.shift_size) > 0:
            H, W, D = self.input_resolution
            img_mask = torch.zeros((1, H, W, D, 1))
            h_slices = (slice(0, -self.window_size[0]),
                        slice(-self.window_size[0], -self.shift_size[0]),
                        slice(-self.shift_size[0], None))
            w_slices = (slice(0, -self.window_size[1]),
                        slice(-self.window_size[1], -self.shift_size[1]),
                        slice(-self.shift_size[1], None))
            d_slices = (slice(0, -self.window_size[2]),
                        slice(-self.window_size[2], -self.shift_size[2]),
                        slice(-self.shift_size[2], None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    for d in d_slices:
                        img_mask[:, h, w, d, :] = cnt
                        cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2])
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, y):
        H, W, D = self.input_resolution
        B, _, C = x.shape
        shortcut = x
        x = x.view(B, H, W, D, C)
        y = y.view(B, H, W, D, C)
        if min(self.shift_size) > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))
            shifted_y = torch.roll(y, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))
        else:
            shifted_x = x
            shifted_y = y
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C)
        y_windows = window_partition(shifted_y, self.window_size)
        y_windows = y_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C)
        attn_windows = self.cross_attn(x_windows, y_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, *self.window_size, C)
        shifted_cross = window_reverse(attn_windows, self.window_size, H, W, D)
        if min(self.shift_size) > 0:
            cross = torch.roll(shifted_cross, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3))
        else:
            cross = shifted_cross
        cross = cross.view(B, H * W * D, C)
        cross = shortcut + self.drop_path(self.norm1(cross))
        cross = cross + self.drop_path(self.norm2(self.mlp(cross)))
        return cross


class PatchMerging(nn.Module):

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        H, W, D = self.input_resolution
        B, _, C = x.shape
        x = x.view(B, H, W, D, C)
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 1::2, 0::2, :]
        x5 = x[:, 0::2, 1::2, 1::2, :]
        x6 = x[:, 1::2, 0::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)
        x = x.view(B, -1, 8 * C)
        x = self.reduction(x)
        x = self.norm(x)
        return x


class BasicLayer(nn.Module):

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.mamba_blocks1 = nn.ModuleList([MambaBlock(dim=dim, norm_layer=norm_layer)] * depth)
        self.mamba_blocks2 = nn.ModuleList([MambaBlock(dim=dim, norm_layer=norm_layer)] * depth)
        self.cross_block1 = CrossSwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                      num_heads=num_heads, window_size=window_size,
                                      shift_size=(0, 0, 0),
                                      mlp_ratio=mlp_ratio,
                                      qkv_bias=qkv_bias,
                                      drop=drop, attn_drop=attn_drop,
                                      drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
                                      norm_layer=norm_layer)
        self.cross_block2 = CrossSwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                      num_heads=num_heads, window_size=window_size,
                                      shift_size=(0, 0, 0),
                                      mlp_ratio=mlp_ratio,
                                      qkv_bias=qkv_bias,
                                      drop=drop, attn_drop=attn_drop,
                                      drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
                                      norm_layer=norm_layer)
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, y):
        H, W, D = self.input_resolution
        B, _, C = x.shape
        for blk in self.mamba_blocks1:
            x = blk(x)
        for blk in self.mamba_blocks2:
            y = blk(y)
        cross_x = self.cross_block1(x, y).permute(0, 2, 1).contiguous().view(B, C, H, W, D)
        cross_y = self.cross_block2(y, x).permute(0, 2, 1).contiguous().view(B, C, H, W, D)
        if self.downsample is not None:
            x = self.downsample(x)
            y = self.downsample(y)
        return x, y, cross_x, cross_y

    def _init_respostnorm(self):
        nn.init.constant_(self.cross_block1.norm1.bias, 0)
        nn.init.constant_(self.cross_block1.norm1.weight, 0)
        nn.init.constant_(self.cross_block1.norm2.bias, 0)
        nn.init.constant_(self.cross_block1.norm2.weight, 0)
        nn.init.constant_(self.cross_block2.norm1.bias, 0)
        nn.init.constant_(self.cross_block2.norm1.weight, 0)
        nn.init.constant_(self.cross_block2.norm2.bias, 0)
        nn.init.constant_(self.cross_block2.norm2.weight, 0)


class ConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size=3, stride=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv3d(in_chans, out_chans, kernel_size=kernel_size, padding=kernel_size // 2, stride=stride),
            nn.InstanceNorm3d(out_chans),
            nn.ReLU(inplace=True))
    
    def forward(self, x):
        return self.main(x)


class ConvPatchEmbedding(nn.Module):

    def __init__(self, in_chans=3, embed_dim=96):
        super().__init__()
        self.main = nn.Sequential(
            ConvBlock(in_chans, 24),
            ConvBlock(24, 48, stride=2),
            ConvBlock(48, 96),
            ConvBlock(96, 192, stride=2),
            ConvBlock(192, embed_dim, kernel_size=1))
    
    def forward(self, x):
        return self.main(x).flatten(2).transpose(1, 2)

class PatchEmbedding(nn.Module):

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x

class SwinTransformer(nn.Module):

    def __init__(self, img_size=(224, 224, 224), in_chans=3,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=(7, 7, 7), mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=.2,
                 norm_layer=nn.LayerNorm, ape=False):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        # self.patch_embed1 = ConvPatchEmbedding(in_chans=in_chans, embed_dim=embed_dim)
        # self.patch_embed2 = ConvPatchEmbedding(in_chans=in_chans, embed_dim=embed_dim)
        # 消融实验
        self.patch_embed1 = PatchEmbedding(in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer)
        self.patch_embed2 = PatchEmbedding(in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer)
        num_patches = img_size[0] * img_size[1] * img_size[2] // 64
        patches_resolution = [img_size[0] // 4, img_size[1] // 4, img_size[2] // 4]
        self.patches_resolution = patches_resolution
        if self.ape:
            self.absolute_pos_embed1 = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed1, std=.02)
            self.absolute_pos_embed2 = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed2, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer),
                                                 patches_resolution[2] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None)
            self.layers.append(layer)

        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed1", "absolute_pos_embed2"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"cpb_mlp", "logit_scale", "relative_position_bias_table"}

    def forward(self, x, y):
        B = x.shape[0]
        x = self.patch_embed1(x)
        y = self.patch_embed2(y)
        if self.ape:
            x = x + self.absolute_pos_embed1
            y = y + self.absolute_pos_embed2
        x = self.pos_drop(x)
        y = self.pos_drop(y)
        out_x = []
        out_y = []
        for layer in self.layers:
            x, y, cross_x, cross_y = layer(x, y)
            out_x.append(cross_x)
            out_y.append(cross_y)
        return out_x, out_y


class DecoderBlock(nn.Module):

    def __init__(self, in_chans, skip_chans, out_chans):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.conv1 = ConvBlock(in_chans + skip_chans, out_chans)
        self.conv2 = ConvBlock(out_chans + skip_chans, out_chans)
        self.conv3 = ConvBlock(out_chans, out_chans)

    def forward(self, cross, cross_x, cross_y):
        cross = self.conv1(torch.cat([self.up(cross), cross_x], dim=1))
        cross = self.conv2(torch.cat([cross, cross_y], dim=1))
        return self.conv3(cross)


class RegistrationHead(nn.Module):

    def __init__(self, in_chans):
        super().__init__()
        self.main = nn.Conv3d(in_chans, 3, kernel_size=3, padding=1)
        self.main.weight = nn.Parameter(Normal(0, 1e-5).sample(self.main.weight.shape))
        self.main.bias = nn.Parameter(torch.zeros(self.main.bias.shape))

    def forward(self, x):
        return self.main(x)


class Final(nn.Module):

    def __init__(self, embed_dim):
        super().__init__()
        self.embdedding_conv1 = nn.Sequential(
            nn.AvgPool3d(kernel_size=2, stride=2),
            ConvBlock(2, embed_dim))
        self.embdedding_conv2 = ConvBlock(2, embed_dim // 2)
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.conv1 = ConvBlock(embed_dim * 2, embed_dim // 2)
        self.conv2 = ConvBlock(embed_dim, embed_dim // 4)
        self.head = RegistrationHead(embed_dim // 4)

    def forward(self, cross, x, y):
        cross = self.conv1(torch.cat([self.up(cross), self.embdedding_conv1(torch.cat([x, y], dim=1))], dim=1))
        cross = self.conv2(torch.cat([self.up(cross), self.embdedding_conv2(torch.cat([x, y], dim=1))], dim=1))
        return self.head(cross)


class SpatialTransformer(nn.Module):

    def __init__(self, img_size, mode="bilinear"):
        super().__init__()
        self.img_size = img_size
        self.mode = mode

    def _get_grid(self):
        vectors = [torch.arange(s) for s in self.img_size]
        grid = torch.stack(torch.meshgrid(vectors, indexing="ij")).unsqueeze(0).float()
        return grid

    def forward(self, x, flow_field):
        locations = self._get_grid().to(x.device) + flow_field
        for i in range(3):
            locations[:, i, ...] = (locations[:, i, ...] / (self.img_size[i] - 1) - 0.5) * 2
        out = F.grid_sample(x, locations.permute(0, 2, 3, 4, 1).contiguous()[..., [2, 1, 0]], mode=self.mode, align_corners=False)
        return out


class PUA(nn.Module):

    def __init__(self, img_size=(224, 224, 224), embed_dim=72, depths=[2, 2, 4, 2], num_heads=[4, 4, 8, 8], window_size=(7, 7, 7)):
    # 消融实验
    # def __init__(self, img_size=(224, 224, 224), embed_dim=48, depths=[2, 2, 4, 2], num_heads=[4, 4, 8, 8], window_size=(7, 7, 7)):
        super().__init__()
        self.transformer = SwinTransformer(img_size=img_size, in_chans=1, embed_dim=embed_dim, 
                                           depths=depths, num_heads=num_heads, window_size=window_size,
                                           drop_path_rate=.3)
        num_layers = len(depths)
        self.decoders = nn.ModuleList([DecoderBlock(embed_dim * 2 ** (num_layers - i_layer - 1), 
                                                    embed_dim // 2 * 2 ** (num_layers - i_layer - 1), 
                                                    embed_dim // 2 * 2 ** (num_layers - i_layer - 1)) for i_layer in range(num_layers - 1)])
        self.final = Final(embed_dim)

    def forward(self, x, y):
        cross = torch.cat([x, y], dim=1)
        history_x, history_y = self.transformer(x, y)
        cross = history_x.pop()
        history_y.pop()
        for decoder in self.decoders:
            cross = decoder(cross, history_x.pop(), history_y.pop())
        return self.final(cross, x, y)
