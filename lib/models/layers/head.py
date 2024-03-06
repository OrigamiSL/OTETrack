import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional
from torch import Tensor
from torch.nn import Identity
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath
from lib.models.layers.frozen_bn import FrozenBatchNorm2d
import copy

def top_k_top_p_filtering_batch(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:

        for i in range(logits.shape[0]):
            indices_to_remove = logits[i] < torch.topk(logits[i], top_k)[0][..., -1, None]
            logits[i][indices_to_remove] = filter_value

    if top_p > 0.0:
        for i in range(logits.shape[0]):
            sorted_logits, sorted_indices = torch.sort(logits[i], descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[i][indices_to_remove] = filter_value
    return logits
    
def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1,
         freeze_bn=False):
    if freeze_bn:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            FrozenBatchNorm2d(out_planes),
            nn.ReLU(inplace=True))
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))
    
class Corner_Predictor(nn.Module):
    """ Corner Predictor module"""

    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16, freeze_bn=False):
        super(Corner_Predictor, self).__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride
        '''top-left corner'''
        self.conv1_tl = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_tl = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_tl = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_tl = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_tl = nn.Conv2d(channel // 8, 1, kernel_size=1)

        '''bottom-right corner'''
        self.conv1_br = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_br = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_br = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_br = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_br = nn.Conv2d(channel // 8, 1, kernel_size=1)

        '''about coordinates and indexs'''
        with torch.no_grad():
            self.indice = torch.arange(0, self.feat_sz).view(-1, 1) * self.stride
            # generate mesh-grid
            self.coord_x = self.indice.repeat((self.feat_sz, 1)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()
            self.coord_y = self.indice.repeat((1, self.feat_sz)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()

    def forward(self, x, return_dist=False, softmax=True):
        """ Forward pass with input x. """
        score_map_tl, score_map_br = self.get_score_map(x)
        if return_dist:
            coorx_tl, coory_tl, prob_vec_tl = self.soft_argmax(score_map_tl, return_dist=True, softmax=softmax)
            coorx_br, coory_br, prob_vec_br = self.soft_argmax(score_map_br, return_dist=True, softmax=softmax)
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz, prob_vec_tl, prob_vec_br
        else:
            coorx_tl, coory_tl = self.soft_argmax(score_map_tl)
            coorx_br, coory_br = self.soft_argmax(score_map_br)
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz

    def get_score_map(self, x):
        # top-left branch
        x_tl1 = self.conv1_tl(x)
        x_tl2 = self.conv2_tl(x_tl1)
        x_tl3 = self.conv3_tl(x_tl2)
        x_tl4 = self.conv4_tl(x_tl3)
        score_map_tl = self.conv5_tl(x_tl4)

        # bottom-right branch
        x_br1 = self.conv1_br(x)
        x_br2 = self.conv2_br(x_br1)
        x_br3 = self.conv3_br(x_br2)
        x_br4 = self.conv4_br(x_br3)
        score_map_br = self.conv5_br(x_br4)
        return score_map_tl, score_map_br

    def soft_argmax(self, score_map, return_dist=False, softmax=True):
        """ get soft-argmax coordinate for a given heatmap """
        score_vec = score_map.view((-1, self.feat_sz * self.feat_sz))  # (batch, feat_sz * feat_sz)
        prob_vec = nn.functional.softmax(score_vec, dim=1)
        exp_x = torch.sum((self.coord_x * prob_vec), dim=1)
        exp_y = torch.sum((self.coord_y * prob_vec), dim=1)
        if return_dist:
            if softmax:
                return exp_x, exp_y, prob_vec
            else:
                return exp_x, exp_y, score_vec
        else:
            return exp_x, exp_y

class CenterPredictor(nn.Module, ):
    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16, freeze_bn=False):
        super(CenterPredictor, self).__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride

        # corner predict
        self.conv1_ctr = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_ctr = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_ctr = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_ctr = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_ctr = nn.Conv2d(channel // 8, 1, kernel_size=1)

        # size regress
        self.conv1_offset = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_offset = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_offset = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_offset = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_offset = nn.Conv2d(channel // 8, 2, kernel_size=1)

        # size regress
        self.conv1_size = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_size = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_size = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_size = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_size = nn.Conv2d(channel // 8, 2, kernel_size=1)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, gt_score_map=None):
        """ Forward pass with input x. """
        score_map_ctr, size_map, offset_map = self.get_score_map(x)

        # assert gt_score_map is None
        if gt_score_map is None:
            bbox = self.cal_bbox(score_map_ctr, size_map, offset_map)
        else:
            bbox = self.cal_bbox(gt_score_map.unsqueeze(1), size_map, offset_map)

        return score_map_ctr, bbox, size_map, offset_map

    def cal_bbox(self, score_map_ctr, size_map, offset_map, return_score=False):
        max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz

        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=idx)
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)

        # bbox = torch.cat([idx_x - size[:, 0] / 2, idx_y - size[:, 1] / 2,
        #                   idx_x + size[:, 0] / 2, idx_y + size[:, 1] / 2], dim=1) / self.feat_sz
        # cx, cy, w, h
        bbox = torch.cat([(idx_x.to(torch.float) + offset[:, :1]) / self.feat_sz,
                          (idx_y.to(torch.float) + offset[:, 1:]) / self.feat_sz,
                          size.squeeze(-1)], dim=1)

        if return_score:
            return bbox, max_score
        return bbox

    def get_pred(self, score_map_ctr, size_map, offset_map):
        max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz

        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=idx)
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)

        # bbox = torch.cat([idx_x - size[:, 0] / 2, idx_y - size[:, 1] / 2,
        #                   idx_x + size[:, 0] / 2, idx_y + size[:, 1] / 2], dim=1) / self.feat_sz
        return size * self.feat_sz, offset

    def get_score_map(self, x):

        def _sigmoid(x):
            y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
            return y

        # ctr branch
        x_ctr1 = self.conv1_ctr(x)
        x_ctr2 = self.conv2_ctr(x_ctr1)
        x_ctr3 = self.conv3_ctr(x_ctr2)
        x_ctr4 = self.conv4_ctr(x_ctr3)
        score_map_ctr = self.conv5_ctr(x_ctr4)

        # offset branch
        x_offset1 = self.conv1_offset(x)
        x_offset2 = self.conv2_offset(x_offset1)
        x_offset3 = self.conv3_offset(x_offset2)
        x_offset4 = self.conv4_offset(x_offset3)
        score_map_offset = self.conv5_offset(x_offset4)

        # size branch
        x_size1 = self.conv1_size(x)
        x_size2 = self.conv2_size(x_size1)
        x_size3 = self.conv3_size(x_size2)
        x_size4 = self.conv4_size(x_size3)
        score_map_size = self.conv5_size(x_size4)
        return _sigmoid(score_map_ctr), _sigmoid(score_map_size), score_map_offset

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, BN=False):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        if BN:
            self.layers = nn.ModuleList(nn.Sequential(nn.Linear(n, k), nn.BatchNorm1d(k))
                                        for n, k in zip([input_dim] + h, h + [output_dim]))
        else:
            self.layers = nn.ModuleList(nn.Linear(n, k)
                                        for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 attn_pos_encoding_only=False):
        super(SelfAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        if attn_pos_encoding_only:
            self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        else:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.k = nn.Linear(dim, dim, bias=qkv_bias)
            self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_pos_encoding_only = attn_pos_encoding_only

    def forward(self, x, q_ape, k_ape, attn_pos):
        '''
            Args:
                x (torch.Tensor): (B, L, C)
                q_ape (torch.Tensor | None): (1 or B, L, C), absolute positional encoding for q
                k_ape (torch.Tensor | None): (1 or B, L, C), absolute positional encoding for k
                attn_pos (torch.Tensor | None): (1 or B, num_heads, L, L), untied positional encoding
            Returns:
                torch.Tensor: (B, L, C)
        '''
        B, N, C = x.shape

        if self.attn_pos_encoding_only:
            assert q_ape is None and k_ape is None
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            q = x + q_ape if q_ape is not None else x
            q = self.q(q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

            k = x + k_ape if k_ape is not None else x
            k = self.k(k).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v = self.v(x).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = q @ k.transpose(-2, -1)
        attn = attn * self.scale
        if attn_pos is not None:
            attn = attn + attn_pos
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 attn_pos_encoding_only=False):
        super(CrossAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        if attn_pos_encoding_only:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv = nn.Linear(dim, 2 * dim, bias=qkv_bias)
        else:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.k = nn.Linear(dim, dim, bias=qkv_bias)
            self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_pos_encoding_only = attn_pos_encoding_only

    def forward(self, q, kv, q_ape, k_ape, attn_pos, get_attention_box = False):
        '''
            Args:
                q (torch.Tensor): (B, L_q, C)
                kv (torch.Tensor): (B, L_kv, C)
                q_ape (torch.Tensor | None): (1 or B, L_q, C), absolute positional encoding for q
                k_ape (torch.Tensor | None): (1 or B, L_kv, C), absolute positional encoding for k
                attn_pos (torch.Tensor | None): (1 or B, num_heads, L_q, L_kv), untied positional encoding
            Returns:
                torch.Tensor: (B, L_q, C)
        '''
        B, q_N, C = q.shape
        kv_N = kv.shape[1]

        if self.attn_pos_encoding_only:
            assert q_ape is None and k_ape is None
            q = self.q(q).reshape(B, q_N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            kv = self.kv(kv).reshape(B, kv_N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]
        else:
            q = q + q_ape if q_ape is not None else q
            q = self.q(q).reshape(B, q_N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            k = kv + k_ape if k_ape is not None else kv
            k = self.k(k).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v = self.v(kv).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = q @ k.transpose(-2, -1)
        attn = attn * self.scale
        if attn_pos is not None:
            attn = attn + attn_pos
        if get_attention_box:
            B, H, L_X, D = q.shape
            x_norm = q / torch.sqrt(torch.sum(q ** 2, dim=-1, keepdim=True)).expand(B, H, L_X, D)
            B, H, L_Z, D = k.shape
            y_norm = k / torch.sqrt(torch.sum(k ** 2, dim=-1, keepdim=True)).expand(B, H, L_Z, D)
            attn_weight = torch.abs(torch.einsum("bvsd,bvpd->bvsp", x_norm, y_norm)) 
            # attn_weight = attn # b * head * q_x * q_z
            # attn_weight = torch.mean(attn_weight,dim = 1) #b * 1 * q_x * q_z
            # attn_weight = attn_weight.squeeze(1) #b * q_x * q_z
            attn_weight = torch.max(attn_weight,dim = 1)[0]
            # print(attn_weight[0,0:10,0:10])
            # exit(-1)
            attn_box = torch.max(attn_weight,dim = 2)[0] #b * q_x
            # print('attn_box', )
            # attn_top = attn_weight.topk(k = n_patches, dim = 2)[1]  #b * head * n_patches,index   
               
        # B, V, P, D = x_enc.shape
        # x_enc_norm = x_enc / torch.sqrt(torch.sum(x_enc ** 2, dim=-1, keepdim=True)).expand(B, V, P, D)
        # B, V, S, D = y.shape
        # y_norm = y / torch.sqrt(torch.sum(y ** 2, dim=-1, keepdim=True)).expand(B, V, S, D)
        # correlation = self.dropout(torch.abs(torch.einsum("bvsd,bvpd->bvsp", y_norm, x_enc_norm)))
        # max_correlation, _ = torch.max(correlation + 1e-6, dim=-1)
        # loss_corr = -torch.log(max_correlation)
        # loss_corr = torch.mean(loss_corr)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, q_N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if not get_attention_box:
            return x
        else:
            return x, attn_box

class CrossAttention_seqz(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 attn_pos_encoding_only=False):
        super(CrossAttention_seqz, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        if attn_pos_encoding_only:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv = nn.Linear(dim, 2 * dim, bias=qkv_bias)
        else:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            # self.q2 = nn.Linear(dim, dim, bias=qkv_bias)
            self.k = nn.Linear(dim, dim, bias=qkv_bias)
            self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_pos_encoding_only = attn_pos_encoding_only

    def forward(self, q, kv, q_ape, k_ape, attn_pos,z_shape, get_attention_box = False):
        '''
            Args:
                q (torch.Tensor): (B, L_q, C)
                kv (torch.Tensor): (B, L_kv, C)
                q_ape (torch.Tensor | None): (1 or B, L_q, C), absolute positional encoding for q
                k_ape (torch.Tensor | None): (1 or B, L_kv, C), absolute positional encoding for k
                attn_pos (torch.Tensor | None): (1 or B, num_heads, L_q, L_kv), untied positional encoding
            Returns:
                torch.Tensor: (B, L_q, C)
        '''
        B, q_N, C = q.shape
        kv_N = kv.shape[1]

        # if self.attn_pos_encoding_only:
        #     assert q_ape is None and k_ape is None
        #     q = self.q(q).reshape(B, q_N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        #     kv = self.kv(kv).reshape(B, kv_N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        #     k, v = kv[0], kv[1]
        # else:
        q = q + q_ape 
        q = self.q(q).reshape(B, q_N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # kv[:,z_shape:,:] += k_ape
        k = kv 
        k[:,z_shape:,:] += k_ape
        k = self.k(k).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        v = self.v(kv).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = q @ k.transpose(-2, -1)
        attn = attn * self.scale
        # print(attn.shape,attn_pos.shape)
        if attn_pos is not None:
            attn[:,:,:,0:z_shape] += attn_pos
    
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, q_N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if not get_attention_box:
            return x
        
class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        '''
            Args:
                x (torch.Tensor): (B, L, C), input tensor
            Returns:
                torch.Tensor: (B, L, C), output tensor
        '''
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class FeatureFusion(nn.Module):
    def __init__(self,
                 dim, num_heads, mlp_ratio=2., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=nn.Identity(), act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_pos_encoding_only=False
                 ,get_attention_box = False):
        super(FeatureFusion, self).__init__()
        self.z_norm1 = norm_layer(dim)
        self.x_norm1 = norm_layer(dim)
        self.z_self_attn = SelfAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, attn_pos_encoding_only)
        self.x_self_attn = SelfAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, attn_pos_encoding_only)

        self.z_norm2_1 = norm_layer(dim)#
        self.z_norm2_2 = norm_layer(dim)
        self.x_norm2_1 = norm_layer(dim)#
        self.x_norm2_2 = norm_layer(dim)

        self.z_x_cross_attention = CrossAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, attn_pos_encoding_only)#
        self.x_z_cross_attention = CrossAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, attn_pos_encoding_only)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.z_norm3 = norm_layer(dim)#
        self.x_norm3 = norm_layer(dim)
        # print(mlp_ratio)
        self.z_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)#
        self.x_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.drop_path = drop_path
        self.get_attention_box = get_attention_box

    def forward(self, z, x, z_self_attn_pos, x_self_attn_pos, z_x_cross_attn_pos, x_z_cross_attn_pos):
        z = z + self.drop_path(self.z_self_attn(self.z_norm1(z), None, None, z_self_attn_pos))
        x = x + self.drop_path(self.x_self_attn(self.x_norm1(x), None, None, x_self_attn_pos))
        if self.get_attention_box:
            attention_box_x = x
            attention_box_z = z
            z = z + self.drop_path(self.z_x_cross_attention(self.z_norm2_1(z), self.x_norm2_1(x), None, None, z_x_cross_attn_pos))#
            x = x + self.drop_path(self.x_z_cross_attention(self.x_norm2_2(x), self.z_norm2_2(z), None, None, x_z_cross_attn_pos))
            z = z + self.drop_path(self.z_mlp(self.z_norm3(z)))#
            x = x + self.drop_path(self.x_mlp(self.x_norm3(x)))
            return z, x,attention_box_x,attention_box_z
        else:
            z = z + self.drop_path(self.z_x_cross_attention(self.z_norm2_1(z), self.x_norm2_1(x), None, None, z_x_cross_attn_pos))#
            x = x + self.drop_path(self.x_z_cross_attention(self.x_norm2_2(x), self.z_norm2_2(z), None, None, x_z_cross_attn_pos))
            z = z + self.drop_path(self.z_mlp(self.z_norm3(z)))#
            x = x + self.drop_path(self.x_mlp(self.x_norm3(x)))
            return z, x
    
        # if not self.get_attention_box:
        #     x = x + self.drop_path(self.x_z_cross_attention(self.x_norm2_2(x), self.z_norm2_2(z), None, None, x_z_cross_attn_pos,self.get_attention_box))
        #     z = z + self.drop_path(self.z_mlp(self.z_norm3(z)))#
        #     x = x + self.drop_path(self.x_mlp(self.x_norm3(x)))
        #     return z, x
        # else:
        #     x_output, attention_box = self.x_z_cross_attention(self.x_norm2_2(x), self.z_norm2_2(z), None, None, x_z_cross_attn_pos,self.get_attention_box)
        #     x = x + self.drop_path(x_output)
        #     z = z + self.drop_path(self.z_mlp(self.z_norm3(z)))#
        #     x = x + self.drop_path(self.x_mlp(self.x_norm3(x)))
        #     return z, x, attention_box

class FeatureFusionEncoder(nn.Module):
    def __init__(self, feature_fusion_layers, z_pos_enc, x_pos_enc,
                 z_rel_pos_index, x_rel_pos_index, z_x_rel_pos_index, x_z_rel_pos_index,
                 z_rel_pos_bias_table, x_rel_pos_bias_table, z_x_rel_pos_bias_table, x_z_rel_pos_bias_table):
        super(FeatureFusionEncoder, self).__init__()
        self.layers = nn.ModuleList(feature_fusion_layers)
        self.z_pos_enc = z_pos_enc
        self.x_pos_enc = x_pos_enc
        self.register_buffer('z_rel_pos_index', z_rel_pos_index, False)
        self.register_buffer('x_rel_pos_index', x_rel_pos_index, False)
        self.register_buffer('z_x_rel_pos_index', z_x_rel_pos_index, False)
        self.register_buffer('x_z_rel_pos_index', x_z_rel_pos_index, False)
        self.z_rel_pos_bias_table = z_rel_pos_bias_table
        self.x_rel_pos_bias_table = x_rel_pos_bias_table
        self.z_x_rel_pos_bias_table = z_x_rel_pos_bias_table
        self.x_z_rel_pos_bias_table = x_z_rel_pos_bias_table

    def forward(self, z, x, z_pos, x_pos):
        '''
            Args:
                z (torch.Tensor): (B, L_z, C), template image feature tokens
                x (torch.Tensor): (B, L_x, C), search image feature tokens
                z_pos (torch.Tensor | None): (1 or B, L_z, C), optional positional encoding for z
                x_pos (torch.Tensor | None): (1 or B, L_x, C), optional positional encoding for x
            Returns:
                Tuple[torch.Tensor, torch.Tensor]:
                    (B, L_z, C): template image feature tokens
                    (B, L_x, C): search image feature tokens
        '''
        # Support untied positional encoding only for simplicity
        assert z_pos is None and x_pos is None

        # untied positional encoding
        z_q_pos, z_k_pos = self.z_pos_enc()
        x_q_pos, x_k_pos = self.x_pos_enc()
        z_self_attn_pos = (z_q_pos @ z_k_pos.transpose(-2, -1)).unsqueeze(0)
        x_self_attn_pos = (x_q_pos @ x_k_pos.transpose(-2, -1)).unsqueeze(0)

        z_x_cross_attn_pos = (z_q_pos @ x_k_pos.transpose(-2, -1)).unsqueeze(0)
        x_z_cross_attn_pos = (x_q_pos @ z_k_pos.transpose(-2, -1)).unsqueeze(0)

        # relative positional encoding
        z_self_attn_pos = z_self_attn_pos + self.z_rel_pos_bias_table(self.z_rel_pos_index)
        x_self_attn_pos = x_self_attn_pos + self.x_rel_pos_bias_table(self.x_rel_pos_index)
        z_x_cross_attn_pos = z_x_cross_attn_pos + self.z_x_rel_pos_bias_table(self.z_x_rel_pos_index)
        x_z_cross_attn_pos = x_z_cross_attn_pos + self.x_z_rel_pos_bias_table(self.x_z_rel_pos_index)

        for layer in self.layers[:-1]:
            z, x = layer(z, x, z_self_attn_pos, x_self_attn_pos, z_x_cross_attn_pos, x_z_cross_attn_pos)
        z, x, attention_box_x,attention_box_z = self.layers[-1](z, x, z_self_attn_pos, x_self_attn_pos, z_x_cross_attn_pos, x_z_cross_attn_pos)
        # z, x, attention_box_x,attention_box_z = self.layers[0](z, x, z_self_attn_pos, x_self_attn_pos, z_x_cross_attn_pos, x_z_cross_attn_pos)
        # for layer in self.layers[1:]:
        #     z, x = layer(z, x, z_self_attn_pos, x_self_attn_pos, z_x_cross_attn_pos, x_z_cross_attn_pos)
         
        return z, x, attention_box_x,attention_box_z 

class Learned2DPositionalEncoder(nn.Module):
    def __init__(self, dim, w, h):
        super(Learned2DPositionalEncoder, self).__init__()
        self.w_pos = nn.Parameter(torch.empty(w, dim))
        self.h_pos = nn.Parameter(torch.empty(h, dim))
        trunc_normal_(self.w_pos, std=0.02)
        trunc_normal_(self.h_pos, std=0.02)

    def forward(self):
        w = self.w_pos.shape[0]
        h = self.h_pos.shape[0]
        return (self.w_pos[None, :, :] + self.h_pos[:, None, :]).view(h * w, -1)

class Untied2DPositionalEncoder(nn.Module):
    def __init__(self, dim, num_heads, w, h, scale=None, with_q=True, with_k=True,get_seprate = False):
        super(Untied2DPositionalEncoder, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.pos = Learned2DPositionalEncoder(dim, w, h)
        self.norm = nn.LayerNorm(dim)
        self.pos_q_linear = None
        self.pos_k_linear = None
        if with_q:
            self.pos_q_linear = nn.Linear(dim, dim)
        if with_k:
            self.pos_k_linear = nn.Linear(dim, dim)

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = scale or head_dim ** -0.5
        self.get_seprate = get_seprate

    def forward(self):
        pos = self.norm(self.pos())
        
        seq_len = pos.shape[0]
        if not self.get_seprate:
            if self.pos_q_linear is not None and self.pos_k_linear is not None:
                pos_q = self.pos_q_linear(pos).view(seq_len, self.num_heads, -1).transpose(0, 1) * self.scale
                pos_k = self.pos_k_linear(pos).view(seq_len, self.num_heads, -1).transpose(0, 1)
                return pos_q, pos_k
            elif self.pos_q_linear is not None:
                pos_q = self.pos_q_linear(pos).view(seq_len, self.num_heads, -1).transpose(0, 1) * self.scale
                return pos_q
            elif self.pos_k_linear is not None:
                pos_k = self.pos_k_linear(pos).view(seq_len, self.num_heads, -1).transpose(0, 1)
                return pos_k
            
            else:
                raise RuntimeError
        else:
            return pos



class DropPathAllocator:
    def __init__(self, max_drop_path_rate, stochastic_depth_decay = True):
        self.max_drop_path_rate = max_drop_path_rate
        self.stochastic_depth_decay = stochastic_depth_decay
        self.allocated = []
        self.allocating = []

    def __enter__(self):
        self.allocating = []

    def __exit__(self, exc_type, exc_val, exc_tb):
        if len(self.allocating) != 0:
            self.allocated.append(self.allocating)
        self.allocating = None
        if not self.stochastic_depth_decay:
            for depth_module in self.allocated:
                for module in depth_module:
                    if isinstance(module, DropPath):
                        module.drop_prob = self.max_drop_path_rate
        else:
            depth = self.get_depth()
            dpr = [x.item() for x in torch.linspace(0, self.max_drop_path_rate, depth)]
            assert len(dpr) == len(self.allocated)
            for drop_path_rate, depth_modules in zip(dpr, self.allocated):
                for module in depth_modules:
                    if isinstance(module, DropPath):
                        module.drop_prob = drop_path_rate

    def __len__(self):
        length = 0

        for depth_modules in self.allocated:
            length += len(depth_modules)

        return length

    def increase_depth(self):
        self.allocated.append(self.allocating)
        self.allocating = []

    def get_depth(self):
        return len(self.allocated)

    def allocate(self):
        if self.max_drop_path_rate == 0 or (self.stochastic_depth_decay and self.get_depth() == 0):
            drop_path_module = Identity()
        else:
            drop_path_module = DropPath()
        self.allocating.append(drop_path_module)
        return drop_path_module

    def get_all_allocated(self):
        allocated = []
        for depth_module in self.allocated:
            for module in depth_module:
                allocated.append(module)
        return allocated



class TargetQueryDecoderLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=nn.Identity(), act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(TargetQueryDecoderLayer, self).__init__()
        self.norm_1 = norm_layer(dim)

        self.self_attn1 = nn.MultiheadAttention(dim, num_heads, dropout=drop)
        self.norm_2_query = norm_layer(dim)
        self.norm_2_memory = norm_layer(dim)

        self.multihead_attn = nn.MultiheadAttention(dim, num_heads, dropout=drop)
        self.norm_3 = norm_layer(dim)
        self.norm_4 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlpz = Mlp(dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.norm_seq = norm_layer(dim)

        self.drop_path = drop_path

    # def forward(self, query, seq_feat, memoryz, memoryx, prev_query_pos, query_pos, pos_z, pos_x, identity, tgt_mask: Optional[Tensor] = None,
    #             memory_mask: Optional[Tensor] = None,
    #             tgt_key_padding_mask: Optional[Tensor] = None,
    #             memory_key_padding_mask: Optional[Tensor] = None,
    #             ):
    def forward(self, query,  memoryxz,  query_pos, pos_xz, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                ):
        # tgt, z_feat, x_feat, pos_z, pos_x, identity, query_embed,
        #                                 tgt_mask=generate_square_subsequent_mask(len(tgt)).to(tgt.device)
        '''
            Args:
                query (torch.Tensor): (B, num_queries, C)
                memory (torch.Tensor): (B, L, C)
                query_pos (torch.Tensor): (1 or B, num_queries, C)
                memory_pos (torch.Tensor): (1 or B, L, C)
            Returns:
                torch.Tensor: (B, num_queries, C)
        '''
        # q2 = self.norm_2_query(query) + query_pos
        # memory = torch.tensor(memoryx)
        # pos =  pos_x
        # ide =  identity[:, 1, :].repeat(1, pos_x.shape[1], 1)
        # k2 = (self.norm_2_memory(memory) + pos + ide).permute(1, 0, 2)
        # memory_in = memory.permute(1, 0, 2)
        # query = query + self.drop_path(
        #     self.multihead_attn(query=q2, key=k2, value=memory_in, attn_mask=memory_mask,
        #                     key_padding_mask=memory_key_padding_mask)[0])
        # query = query + self.drop_path(self.mlpz(self.norm_3(query)))
        
        # # query_self_attention
        # tgt = query
        # q = k = self.norm_4(query) + query_pos
        # query = query + self.drop_path(self.self_attn1(q, k, value=tgt, attn_mask=None,
        #      key_padding_mask=tgt_key_padding_mask)[0])
         
        # return query
        # tgt = query
        # q = k = self.norm_1(query) + query_pos
        # query = query + self.drop_path(self.self_attn1(q, k, value=tgt, attn_mask=tgt_mask,
        #                                                key_padding_mask=None)[0])
        # print('shape1')
        # print(seq_feat.shape,prev_query_pos.shape,memoryx.shape,pos_x.shape)
        q2 = self.norm_2_query(query) + query_pos
        
        memory = memoryxz

        # k2 = torch.cat([(self.norm_seq(seq_feat)+prev_query_pos.permute(1,0,2)),(self.norm_2_memory(memory) + pos + ide)],dim=1).permute(1, 0, 2)
        k2 = (self.norm_2_memory(memory) + pos_xz).permute(1, 0, 2)
        # print('k2',k2.shape)
        # memory_in = torch.cat([seq_feat,memory],dim=1).permute(1, 0, 2)
        memory_in = memory.permute(1, 0, 2)
        # print('memory',memory_in.shape)
        query = query + self.drop_path(
            self.multihead_attn(query=q2, key=k2, value=memory_in, attn_mask=memory_mask,
                            key_padding_mask=memory_key_padding_mask)[0])
        query = query + self.drop_path(self.mlpz(self.norm_3(query)))

        query = self.norm_seq(query)

        return query
    
class QueryAttentionLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=nn.Identity(), act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(QueryAttentionLayer, self).__init__()

        self.self_attn1 = nn.MultiheadAttention(dim, num_heads, dropout=drop)

        self.norm_3 = norm_layer(dim)
        self.norm_4 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlpz = Mlp(dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.norm_seq = norm_layer(dim)

        self.drop_path = drop_path

    def forward(self, query,  query_pos, tgt_mask: Optional[Tensor] = None,
                ):
        
        # # query_self_attention
        tgt = query
        q = k = self.norm_4(query) + query_pos
        query = query + self.drop_path(self.self_attn1(q, k, value=tgt, attn_mask=tgt_mask,
             key_padding_mask=None)[0])
         
        query = query + self.drop_path(self.mlpz(self.norm_3(query)))

        query = self.norm_seq(query)

        return query

class TargetQueryDecoderLayer_2xfeat(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=nn.Identity(), act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(TargetQueryDecoderLayer_2xfeat, self).__init__()
        self.norm_1 = norm_layer(dim)

        self.self_attn1 = nn.MultiheadAttention(dim, num_heads, dropout=drop)
        self.norm_2_query = norm_layer(dim)
        self.norm_2_memory = norm_layer(dim)

        self.multihead_attn = nn.MultiheadAttention(dim, num_heads, dropout=drop)
        self.norm_3 = norm_layer(dim)
        self.norm_4 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlpz = Mlp(dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.norm_seq = norm_layer(dim)

        self.drop_path = drop_path

    def forward(self, query,  memoryz, memoryx, memoryx_seq, query_pos, pos_z, pos_x, pos_x_seq, identity, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                ):
       
        q2 = self.norm_2_query(query) + query_pos
        # q2 = query
        # print('q2',q2.shape)

        memory = torch.cat((memoryz,memoryx,memoryx_seq),dim=1)
        pos = torch.cat((pos_z, pos_x, pos_x_seq), dim=1)

        # memory = torch.cat((memoryz,memoryx_seq),dim=1)
        # pos = torch.cat((pos_z,pos_x_seq), dim=1)
        
        ide = torch.cat((identity[: ,0, :].repeat(1, pos_z.shape[1], 1), identity[:, 1, :].repeat(1, pos_x.shape[1], 1)), dim=1)

        k_before = self.norm_2_memory(memory)
        k_before[:,:-pos_x.shape[1],:] += ide

        # print('k_before',k_before.shape)

        k2 = (k_before + pos).permute(1, 0, 2)

        memory_in = memory.permute(1, 0, 2)

        query = query + self.drop_path(
            self.multihead_attn(query=q2, key=k2, value=memory_in, attn_mask=memory_mask,
                            key_padding_mask=memory_key_padding_mask)[0])
        query = query + self.drop_path(self.mlpz(self.norm_3(query)))

        return query
    
class TargetQueryDecoderLayer_selfatt(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=nn.Identity(), act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(TargetQueryDecoderLayer_selfatt, self).__init__()
        self.norm_1 = norm_layer(dim)

        # self.self_attn1 = nn.MultiheadAttention(dim, num_heads, dropout=drop)
        self.norm_2_query = norm_layer(dim)
        self.norm_2_memory = norm_layer(dim)

        self.multihead_attn = nn.MultiheadAttention(dim, num_heads, dropout=drop)
        self.norm_3 = norm_layer(dim)
        self.norm_4 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlpz = Mlp(dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.seq_att1= SeqAttentionLayer(dim, num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop)
        # self.seq_att2= SeqAttentionLayer(dim, num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop)

        self.norm_seq = norm_layer(dim)

        self.drop_path = drop_path

    def forward(self, seq_x1,seq_y1,seq_x2,seq_y2,x1_embed,y1_embed,x2_embed,y2_embed,
                query,  memoryz, memoryx, memoryx_seq, query_pos, pos_z, pos_x, pos_x_seq, identity, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                tgt_seq_mask: Optional[Tensor] = None,
                use_seq = False,
                ):

        # seq_x1,seq_y1,seq_x2,seq_y2,x1_embed,y1_embed,x2_embed,y2_embed,tgt_emd,
        #         tgt_mask: Optional[Tensor] = None,
        #         tgt_key_padding_mask: Optional[Tensor] = None,
        # print('query',query.shape,seq_x1.shape,seq_y1.shape,seq_x2.shape,seq_y2.shape)
        # if use_seq:
        query, x1, y1, x2, y2 = self.seq_att1(query,seq_x1,seq_y1,seq_x2,seq_y2,x1_embed,y1_embed,x2_embed,y2_embed,query_pos,
                tgt_mask = tgt_mask,
                tgt_seq_mask = None)
        # x1, y1, x2, y2 = 0,0,0,0
        # q2 = self.norm_2_query(query) + query_pos
       
        # memory = torch.cat((memoryz,memoryx),dim=1)
        # pos = torch.cat((pos_z, pos_x), dim=1)
       
        # k_before = self.norm_2_memory(memory)
        # # print('k_before',k_before.shape)
        # k2 = (k_before + pos).permute(1, 0, 2)

        # memory_in = memory.permute(1, 0, 2)

        # query = query + self.drop_path(
        #     self.multihead_attn(query=q2, key=k2, value=memory_in, attn_mask=memory_mask,
        #                     key_padding_mask=memory_key_padding_mask)[0])
        # query = query + self.drop_path(self.mlpz(self.norm_3(query)))

        query = self.norm_seq(query)

        # query = self.seq_att2(query,seq_x1,seq_y1,seq_x2,seq_y2,x1_embed,y1_embed,x2_embed,y2_embed,query_pos,
        #         tgt_mask = None,
        #         tgt_seq_mask = tgt_seq_mask)
        return query,x1,y1,x2,y2
    
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TargetQueryDecoderBlock(nn.Module):
    def __init__(self, dim, decoder_layers, num_layer,query_attention_layers = None,selfatt=False):
        super(TargetQueryDecoderBlock, self).__init__()
        self.layers = nn.ModuleList(decoder_layers)
        self.num_layers = num_layer
        self.norm = nn.LayerNorm(dim)
        self.query_attention_layers = query_attention_layers

    # tgt, seq_feat,z_feat, x_feat, pos_z, pos_x, identity, query_embed,prev_query_embed,
    # def forward(self, tgt, seq_feat, z, x, pos_z, pos_x, identity, prev_query_embed, query_pos: Optional[Tensor] = None,
    #             tgt_mask: Optional[Tensor] = None,
    #             memory_mask: Optional[Tensor] = None,
    #             tgt_key_padding_mask: Optional[Tensor] = None,
    #             memory_key_padding_mask: Optional[Tensor] = None):
    # tgt, x_feat_seq, z_feat, x_feat, pos_z, pos_x, self.pos_x_seq, identity, query_embed,
    #                                         tgt_mask= None
    # def forward(self, tgt, x_feat_seq,z, x, pos_z, pos_x, identity, query_pos: Optional[Tensor] = None,
    #             tgt_mask: Optional[Tensor] = None,
    #             memory_mask: Optional[Tensor] = None,
    #             tgt_key_padding_mask: Optional[Tensor] = None,
    #             memory_key_padding_mask: Optional[Tensor] = None):
        
    #query,seq_x1,seq_y1,seq_x2,seq_y2,x1_embed,y1_embed,x2_embed,y2_embed,query_pos,
    # tgt_mask = None,
    # tgt_seq_mask = None
    
    def forward(self, 
                tgt, zx_feat, pos_embed_total,  query_pos: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                tgt_mask_onlyseq: Optional[Tensor] = None):
        '''
            Args:
                z (torch.Tensor): (B, L_z, C)
                x (torch.Tensor): (B, L_x, C)
            Returns:
                torch.Tensor: (B, num_queries, C)
        '''
        # seq_x1,seq_y1,seq_x2,seq_y2,x1_embed,y1_embed,x2_embed,y2_embed,
        #         query,  memoryz, memoryx, memoryx_seq, query_pos, pos_z, pos_x, pos_x_seq, identity, tgt_mask: Optional[Tensor] = None,
        #         memory_mask: Optional[Tensor] = None,
        #         tgt_key_padding_mask: Optional[Tensor] = None,
        #         memory_key_padding_mask: Optional[Tensor] = None,
        #         tgt_seq_mask: Optional[Tensor] = None
        output = tgt
        
        for i in range(len(self.layers)):
                # print('1')
            layer = self.layers[i]
            # if i == 1:
            #     output = self.query_attention_layers(output,query_pos,tgt_mask=tgt_mask)
            output = layer(output,zx_feat, query_pos, pos_embed_total,
                    tgt_mask=tgt_mask,
                    memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask)
            
        output = self.norm(output)

        return output

def build_decoder(decoder_layer, drop_path, dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate, z_size, x_size,two_xfeat = False,selfatt_seq = False):
    z_shape = [z_size, z_size]
    x_shape = [x_size, x_size]
    num_layers = decoder_layer
    # query_attention_layers = QueryAttentionLayer(dim, num_heads, mlp_ratio, qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
    #                                              drop_path=drop_path.allocate())
    query_attention_layers = None
    # self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
    #              drop_path=nn.Identity()
    # query,  query_pos, tgt_mask: Optional[Tensor] = None,
    decoder_layers = []
    if not selfatt_seq:
        for _ in range(num_layers):
            decoder_layers.append(
                TargetQueryDecoderLayer(dim, num_heads, mlp_ratio, qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                                        drop_path=drop_path.allocate()))
            drop_path.increase_depth()
    else:
        for _ in range(num_layers):
            decoder_layers.append(
                TargetQueryDecoderLayer_selfatt(dim, num_heads, mlp_ratio, qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                                        drop_path=drop_path.allocate()))
            drop_path.increase_depth()

    decoder = TargetQueryDecoderBlock(dim, decoder_layers, num_layers, query_attention_layers= query_attention_layers)
    return decoder

def generate_square_subsequent_mask(sz):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float(
        '-inf')).masked_fill(mask == 1, float(0.0))
    return mask
def build_attention_layer(attention_layer, drop_path, dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate, z_size,
                  x_size):
    num_layers = attention_layer
    seq_attention_layers = []
    
    for i in range(num_layers):
        
        seq_attention_layers.append(
            SeqAttentionLayer(dim, num_heads, mlp_ratio, qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                                        drop_path=drop_path.allocate()))
                
        drop_path.increase_depth()
   
    attention = SeqAttentionBlock(dim, seq_attention_layers,num_layers)
    return attention

class SeqAttentionLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=nn.Identity(), act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(SeqAttentionLayer, self).__init__()
        self.norm_1 = norm_layer(dim)
        self.self_attn1 = nn.MultiheadAttention(dim, num_heads, dropout=drop)
        self.norm_2_query = norm_layer(dim)
        self.norm_2_memory = norm_layer(dim)
        self.multihead_attn = nn.MultiheadAttention(dim, num_heads, dropout=drop)
        self.norm_3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlpz = Mlp(dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.norm_all = norm_layer(dim)
        self.norm_output = norm_layer(dim)

        self.drop_path = drop_path
        self.norm_selfattention = norm_layer(dim)

    def get_attention_result(self, seq, embed,tgt_mask,tgt_key_padding_mask=None):
        
        # exit(-1)
        len_seq = seq.shape[0] - 1
        tgt = seq
        seq1 = self.norm_1(seq)
        seq1[:-1,:,:] = seq1[:-1,:,:] + embed[:len_seq]
        # q = k = self.norm_1(seq) + embed
        q = k = seq1
        # start_time = time.time()
        # print('seq',q.shape,k.shape,tgt.shape,tgt_mask.shape)
        seq = seq + self.drop_path(self.self_attn1(q, k, value=tgt, attn_mask=tgt_mask,
                                                       key_padding_mask=tgt_key_padding_mask)[0])
        return seq
    
    def get_attention_result_x1y1x2y2(self, query, seq_x1,seq_y1,seq_x2,seq_y2,x1_embed,y1_embed,x2_embed,y2_embed,tgt_mask,tgt_emd):
        # mask = generate_mask_x1y1x2y2(seq_x1.shape[0]).to(seq_x1.device)
        # print(mask)
        len_seq = seq_x1.shape[0]
        seq_x1 = torch.cat([seq_x1,query[0,:,:].unsqueeze(0)],dim=0)
        seq_y1 = torch.cat([seq_y1,query[1,:,:].unsqueeze(0)],dim=0)
        seq_x2 = torch.cat([seq_x2,query[2,:,:].unsqueeze(0)],dim=0)
        seq_y2 = torch.cat([seq_y2,query[3,:,:].unsqueeze(0)],dim=0)

        seq_all = torch.cat([seq_x1,seq_y1,seq_x2,seq_y2],dim = 0)

        tgt = seq_all

        seq_all = self.norm_all(seq_all)

        # print('len',len_seq )
        # print('seq_all',seq_all.shape)
        seq_all[len_seq,:,:] += tgt_emd[0,:,:]
        seq_all[2*(len_seq+1)-1,:,:] += tgt_emd[1,:,:]
        seq_all[3*(len_seq+1)-1,:,:] += tgt_emd[2,:,:]
        seq_all[4*(len_seq+1)-1,:,:] += tgt_emd[3,:,:]

        seq_all[0:len_seq,:,:] += x1_embed
        seq_all[len_seq+1:2*(len_seq+1)-1,:,:] += y1_embed
        seq_all[2*(len_seq+1):3*(len_seq+1)-1,:,:] += x2_embed
        seq_all[3*(len_seq+1):4*(len_seq+1)-1,:,:] += y2_embed

        q = k = seq_all
       
        tgt = tgt + self.drop_path(self.self_attn1(q, k, value=tgt, attn_mask=tgt_mask,
                                                       key_padding_mask=None)[0])
        # self.norm_output
  
        return tgt

    def forward(self,query,seq_x1,seq_y1,seq_x2,seq_y2,x1_embed,y1_embed,x2_embed,y2_embed,query_pos,
                tgt_mask = None,
                tgt_seq_mask = None
                ):
        # seq: b * 7+1 *c
        # l,b,c
        # if tgt_mask != None :
            seq = self.get_attention_result_x1y1x2y2(query,seq_x1,seq_y1,seq_x2,seq_y2,x1_embed,y1_embed,x2_embed,y2_embed,tgt_mask,query_pos)
            l = int(seq.shape[0]/4)
  
            seq = seq + self.drop_path(self.mlpz(self.norm_3(seq)))

            seq = self.norm_selfattention(seq)
            
            x1 = seq[0:l,:,:]
            y1 = seq[l:2*l,:,:]
            x2 = seq[2*l:3*l,:,:]
            y2 = seq[3*l:4*l,:,:]

            query_x1 = x1[-1,:,:]
            query_y1 = y1[-1,:,:]
            query_x2 = x2[-1,:,:]
            query_y2 = y2[-1,:,:]

            x1 = x1[:-1,:,:]
            y1 = y1[:-1,:,:]
            x2 = x2[:-1,:,:]
            y2 = y2[:-1,:,:]

            query = torch.cat([query_x1.unsqueeze(0),query_y1.unsqueeze(0),query_x2.unsqueeze(0),query_y2.unsqueeze(0)],dim=0)

            return query,x1,y1,x2,y2
        
        # else:

        #     seq = self.get_attention_result_x1y1x2y2(seq_x1,seq_y1,seq_x2,seq_y2,x1_embed,y1_embed,x2_embed,y2_embed,tgt_seq_mask)
        #     l = int(seq.shape[0]/4)
  
        #     seq = seq + self.drop_path(self.mlpz(self.norm_3(seq)))

        #     seq = self.norm_selfattention(seq)
            
        #     x1 = seq[0:l,:,:]
        #     y1 = seq[l:2*l,:,:]
        #     x2 = seq[2*l:3*l,:,:]
        #     y2 = seq[3*l:4*l,:,:]

        #     return x1,y1,x2,y2

class SeqAttentionBlock(nn.Module):
    def __init__(self, dim, seq_attention_layers, num_layer):
        super(SeqAttentionBlock, self).__init__()
        
        self.seq_attention_layers = nn.ModuleList(seq_attention_layers)
        self.num_layers = num_layer
        self.norm = nn.LayerNorm(dim)  

        # act_layer=nn.GELU
        # mlp_ratio = 4
        # drop = 0.1
        # mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp_cross =  Mlp(dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # self.norm_cross = nn.LayerNorm(dim)
        linear_layers = []
        for i in range(0,num_layer -1):
            linear_layers.append(nn.Linear(dim, dim))
        self.linear_layers = nn.ModuleList(linear_layers)

        norm_layers = []
        for i in range(0,num_layer -1):
            norm_layers.append(nn.LayerNorm(dim))
        self.norm_layers = nn.ModuleList(norm_layers)

        # self.norm_x1 = nn.LayerNorm(dim)
        # self.norm_y1 = nn.LayerNorm(dim)
        # self.norm_x2 = nn.LayerNorm(dim)
        # self.norm_y2 = nn.LayerNorm(dim)  
        
    def forward(self, tgt,
                seq_x1,seq_y1,seq_x2,seq_y2,x1_embed,y1_embed,x2_embed,y2_embed,
                query_pos: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None):

        output = tgt
        previous_feat = tgt
        # previous_feat = self.mlp_cross(previous_feat)
        x1,y1,x2,y2 = seq_x1,seq_y1,seq_x2,seq_y2
        # print(x1.shape,output.shape)
        x1  = torch.cat([x1,output[0,:,:].unsqueeze(0)],dim=0)
        y1  = torch.cat([y1,output[1,:,:].unsqueeze(0)],dim=0)
        x2  = torch.cat([x2,output[2,:,:].unsqueeze(0)],dim=0)
        y2  = torch.cat([y2,output[3,:,:].unsqueeze(0)],dim=0)

        for i in range(self.num_layers):
            output_seq_attention_x1, output_seq_attention_y1,output_seq_attention_x2,output_seq_attention_y2\
            = self.seq_attention_layers[i](x1,y1,x2,y2,
                                            x1_embed,y1_embed,x2_embed,y2_embed,
                                            tgt_mask = tgt_mask,
                                            tgt_key_padding_mask=tgt_key_padding_mask)
        
            if i == self.num_layers - 1:

                x1_in,y1_in,x2_in,y2_in =\
                output_seq_attention_x1[-1,:,:].unsqueeze(0), output_seq_attention_y1[-1,:,:].unsqueeze(0),output_seq_attention_x2[-1,:,:].unsqueeze(0),output_seq_attention_y2[-1,:,:].unsqueeze(0)
                # 4 ,b, c
                output_seq_attention = torch.cat([x1_in,y1_in,x2_in,y2_in],dim = 0)
                
                output =  output_seq_attention
            else:

                output_seq_attention_cat = \
                    torch.cat([output_seq_attention_x1,output_seq_attention_y1,output_seq_attention_x2,output_seq_attention_y2],dim = 0)
                
                # output_seq_attention_cat = self.norm_layers[i](output_seq_attention_cat)

                l = int(output_seq_attention_x1.shape[0])
                x1 = output_seq_attention_cat[0:l,:,:]
                y1 = output_seq_attention_cat[l:2*l,:,:]
                x2 = output_seq_attention_cat[2*l:3*l,:,:]
                y2 = output_seq_attention_cat[3*l:4*l,:,:]
                               
        output = self.norm(output)
        return output


def generate_mask_x1y1x2y2(sz):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    matrix = torch.zeros(4 * sz, 4 * sz).float()

    # mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)

    matrix[0:sz,0:sz] = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    matrix[sz:2*sz,sz:2*sz] = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    matrix[2*sz:3*sz,2*sz:3*sz] = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    matrix[3*sz:4*sz,3*sz:4*sz] = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)

    matrix = matrix.float().masked_fill(matrix == 0, float(
        '-inf')).masked_fill(matrix == 1, float(0.0))
    # print(matrix)
    return matrix   
class plus_FeatureFusion(nn.Module):
    def __init__(self,
                 dim, num_heads, mlp_ratio=2., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=nn.Identity(), act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_pos_encoding_only=False
                 ,get_attention_box = False):
        super(plus_FeatureFusion, self).__init__()
        self.zx_norm1 = norm_layer(dim)
        self.zy_norm1 = norm_layer(dim)
        self.x_norm1 = norm_layer(dim)
        self.zx_self_attn = SelfAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, False)
        self.zy_self_attn = SelfAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, False)

        self.x_self_attn = SelfAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, attn_pos_encoding_only)

        self.z_norm2_1 = norm_layer(dim)#
        self.z_norm2_2 = norm_layer(dim)
        self.x_norm2_1 = norm_layer(dim)#
        self.x_norm2_2 = norm_layer(dim)

        self.z_x_cross_attention = CrossAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, False)#
        self.x_z_cross_attention = CrossAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, False)

        # self.x_zseq_cross_attention = CrossAttention_seqz(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, False)
        # self.x_zseq_cross_attention = CrossAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, False)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.z_norm3 = norm_layer(dim)#
        self.x_norm3 = norm_layer(dim)
        # print(mlp_ratio)
        self.z_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)#
        self.x_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.drop_path = drop_path
        self.get_attention_box = get_attention_box

    def forward(self, input_seq, z_template,x, seq_x_pos, seq_y_pos, x_self_attn_pos,x_sep_abs_encoder,z_sep_abs_encoder):
        x_cross_pos = x_sep_abs_encoder.expand(input_seq.shape[0],x_sep_abs_encoder.shape[0],x_sep_abs_encoder.shape[1])
        # z_cross_pos = z_sep_abs_encoder.expand(input_seq.shape[0],z_sep_abs_encoder.shape[0],z_sep_abs_encoder.shape[1])
        lenth = seq_x_pos.shape[1]
        zx = input_seq[:,0:lenth,:]
        zy = input_seq[:,lenth:,:]
        # print(input_seq.shape, x.shape, seq_x_pos.shape, seq_y_pos.shape, x_self_attn_pos.shape,x_sep_abs_encoder.shape)
        # exit(-1)
        zx = zx + self.drop_path(self.zx_self_attn(self.zx_norm1(zx), seq_x_pos, seq_x_pos, None))
        zy = zy + self.drop_path(self.zy_self_attn(self.zy_norm1(zy), seq_y_pos, seq_y_pos, None))
        x = x + self.drop_path(self.x_self_attn(self.x_norm1(x), None, None, x_self_attn_pos))

        z = torch.concat([zx,zy],dim=1)  
        seq_xz_pos = torch.concat([seq_x_pos, seq_y_pos],dim=1)
        # seq_xz_pos = torch.concat([z_cross_pos,seq_x_pos, seq_y_pos],dim=1)

        # zseq = torch.cat([z_template,z],dim=1)
        # z_shape = z_template.shape[1]
        z = z + self.drop_path(self.z_x_cross_attention(self.z_norm2_1(z), self.x_norm2_1(x), seq_xz_pos, x_cross_pos, False))#
        # q, kv, q_ape, k_ape, attn_pos,z_shape, get_attention_box = False
        # x = x + self.drop_path(self.x_zseq_cross_attention(self.x_norm2_2(x), self.z_norm2_2(zseq), x_cross_pos, seq_xz_pos, x_z_cross_attn_pos,z_shape))
        # x = x + self.drop_path(self.x_zseq_cross_attention(self.x_norm2_2(x), self.z_norm2_2(zseq), x_cross_pos, seq_xz_pos,False))
        x = x + self.drop_path(self.x_z_cross_attention(self.x_norm2_2(x), self.z_norm2_2(z), x_cross_pos, seq_xz_pos,False))

        z = z + self.drop_path(self.z_mlp(self.z_norm3(z)))
        x = x + self.drop_path(self.x_mlp(self.x_norm3(x)))
        return z, x
    
def get_sinusoid_encoding_table(n_position, d_hid, cls_token=False):
    ''' Sinusoid position encoding table '''

    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    # pos_embed = torch.FloatTensor(sinusoid_table).unsqueeze(0)
    pos_embed = sinusoid_table
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, d_hid]), pos_embed], axis=0)
    return pos_embed

class Pix2Track(nn.Module):
    def __init__(self, in_channel=64, feat_sz=20, feat_tz=10, range=2, pre_number=3,stride=16, encoder_layer=3, decoder_layer=3,
                 bins=400,num_heads=12, mlp_ratio=2, qkv_bias=True, drop_rate=0.0,attn_drop=0.0, drop_path=nn.Identity):
        super(Pix2Track, self).__init__()

        vit_dim = 768
        in_channel = 256
        self.bins = bins
        self.range = range
        self.word_embeddings = nn.Embedding(self.bins * self.range + 2, in_channel, padding_idx=self.bins * self.range, max_norm=1, norm_type=2.0)
        self.embeddings_norm = nn.LayerNorm(in_channel)
        self.embeddings_drop = nn.Dropout(drop_rate)

        # self.word_embeddings1 = nn.Embedding(self.bins * self.range + 2, in_channel, padding_idx=self.bins * self.range, max_norm=1, norm_type=2.0)
        # self.embeddings_norm1 = nn.LayerNorm(in_channel)
        # self.embeddings_drop1 = nn.Dropout(drop_rate)

        self.pre_number = pre_number
        print('self.pre_number',self.pre_number)
        # print(self.bins)
        # self.position_embeddings = nn.Embedding(
        #     5, in_channel)
        self.position_embeddings = nn.Embedding(
            4, in_channel)
        
        # self.position_embeddings_seq = nn.Embedding(
        #     4, in_channel)
        
        # self.previous_position_embeddings = nn.Embedding(
        #     4 * self.pre_number, in_channel)

        # self.pos_x_seq = Untied2DPositionalEncoder(in_channel, num_heads,feat_sz, feat_sz,get_seprate= True)
        # self.pos_z_seq = Untied2DPositionalEncoder(in_channel, num_heads,feat_tz, feat_tz,get_seprate= True)
        self.output_bias = torch.nn.Parameter(torch.zeros(self.bins * self.range + 2))
        # self.output_bias_seq = torch.nn.Parameter(torch.zeros(self.bins * self.range + 2))

        self.bottleneck = nn.Linear(vit_dim, in_channel)

        # self.norm = nn.LayerNorm(in_channel)
        self.encoder_layer = encoder_layer
        self.drop_path = drop_path
        self.tz = feat_tz * feat_tz
        self.sz = feat_sz * feat_sz
        self.tz_next = (feat_tz-1) * (feat_tz-1)
        self.sz_next = (feat_sz-1) * (feat_sz-1)

        trunc_normal_(self.word_embeddings.weight, std=.02)

        # self.pos_embed_x = nn.Parameter(torch.zeros(1, self.sz, in_channel))
        # pos_embed_x = get_sinusoid_encoding_table(self.sz, self.pos_embed_x.shape[-1], cls_token=False)
        # self.pos_embed_x.data.copy_(torch.from_numpy(pos_embed_x).float().unsqueeze(0))

        # self.pos_embed_z = nn.Parameter(torch.zeros(1, self.tz, in_channel))
        # pos_embed_z = get_sinusoid_encoding_table(self.tz, self.pos_embed_z.shape[-1], cls_token=False)
        # self.pos_embed_z.data.copy_(torch.from_numpy(pos_embed_z).float().unsqueeze(0))
        total_size = self.sz+self.sz_next+self.tz+self.tz_next+self.tz+self.tz_next
        self.pos_embed_total = nn.Parameter(torch.zeros(1, total_size, in_channel))
        pos_embed_total = get_sinusoid_encoding_table(total_size, self.pos_embed_total.shape[-1], cls_token=False)
        self.pos_embed_total.data.copy_(torch.from_numpy(pos_embed_total).float().unsqueeze(0))

        # if self.encoder_layer > 0 :
        #     self.encoder = build_encoder(encoder_layer, num_heads, mlp_ratio, qkv_bias,
        #                 drop_rate, attn_drop, in_channel, feat_tz, feat_sz, self.drop_path)
        # else:
        #     self.encoder = None
        decoder_layer = 4
        self.decoder = build_decoder(decoder_layer, self.drop_path, in_channel, num_heads,
                                     mlp_ratio, qkv_bias, drop_rate, attn_drop, feat_tz, feat_sz)
        
        # decoder_layer_back = 3
        # self.decoder_back = build_decoder(decoder_layer_back, self.drop_path, in_channel, num_heads,
        #                              mlp_ratio, qkv_bias, drop_rate, attn_drop, feat_tz, feat_sz)
        
        # self.decoder_2feat = build_decoder(decoder_layer, self.drop_path, in_channel, num_heads,
        #                              mlp_ratio, qkv_bias, drop_rate, attn_drop, feat_tz, feat_sz,two_xfeat= True)
        
        # self.attention_layer_number = 3
        # self.attention_layer = build_attention_layer(self.attention_layer_number, self.drop_path, in_channel, num_heads,
        #                              mlp_ratio, qkv_bias, drop_rate, attn_drop, feat_tz, feat_sz)

        # decoder_seq_layer = 1
        # self.decoder_seq = build_decoder(decoder_seq_layer, self.drop_path, in_channel, num_heads,
        #                              mlp_ratio, qkv_bias, drop_rate, attn_drop, feat_tz, feat_sz, selfatt_seq = True)
        
        # self.plus_encoder_layer = 3
        # self.plus_encoder = build_plus_encoder(self.plus_encoder_layer, num_heads, mlp_ratio, qkv_bias,
        #                 drop_rate, attn_drop, in_channel, feat_tz, feat_sz, self.drop_path)
    
    def forward(self, zx_feat=None, pos_z=None, pos_x=None, identity=None, seqs_input=None,seq = None, seq_emd = None,flag = 0):
        
        share_weight = self.word_embeddings.weight.T

        # bs = zx_feat.shape[0]

        if flag == 0:
            # share_weight = self.word_embeddings.weight.T

            zx_feat = self.bottleneck(zx_feat)

            # print('zx',zx_feat.shape)

            # print('zx_feat',zx_feat.shape)
            # z_feat = zx_feat[:, :self.tz]
            # x_feat = zx_feat[:, self.tz:]

            bs = zx_feat.shape[0]

            # if self.encoder != None:
            #     # z_feat, x_feat = self.encoder(z_feat, x_feat, None, None)
            #     z_feat, x_feat, attention_map_x, attention_map_z = self.encoder(z_feat, x_feat, None, None)
            #     # z_feat, x_feat, attention_map_x, attention_map_z = self.encoder(z_feat, x_feat, None, None)
            #     # pred_attention_box: b * 256
            #     # print('x_feat',x_feat.shape)
            if seqs_input == None:
                test = False
                # seqs_input = seqs_input.to(torch.int64).to(zx_feat.device)
                origin_seq = torch.ones(bs, 4) * self.bins * self.range
                seqs_input = origin_seq
                seqs_input = seqs_input.to(zx_feat.device).to(torch.int64)

                tgt = self.word_embeddings(seqs_input).permute(1, 0, 2)
                query_embed = self.position_embeddings.weight.unsqueeze(1)
                query_embed = query_embed.repeat(1, bs, 1)
                
                decoder_feat = self.decoder(tgt, z_feat, x_feat, pos_z, pos_x, identity, query_embed,
                                            tgt_mask=generate_square_subsequent_mask(len(tgt)).to(tgt.device))
                # decoder_feat = self.decoder(tgt, None, x_feat, None, pos_x, identity, query_embed,
                #                             tgt_mask=None)
                at = torch.matmul(decoder_feat, share_weight)
                at = at + self.output_bias    
                output = {'feat': at, "state": "train", "attention_map_x":attention_map_x,'attention_map_z':attention_map_z}
            else:
                # print('strat_test')
                origin_seq = 0.5 * torch.ones(bs, 4) * self.bins * self.range
                origin_seq = origin_seq.to(zx_feat.device).to(torch.int64)
                tgt = self.embeddings_drop(self.embeddings_norm(self.word_embeddings(origin_seq))).permute(1, 0, 2)

                # origin_seq1= 0.5 * torch.ones(bs, 4) * self.bins * self.range
                # origin_seq1 = origin_seq1.to(zx_feat.device).to(torch.int64)
                # tgt1 = self.embeddings_drop1(self.embeddings_norm1(self.word_embeddings1(origin_seq1))).permute(1, 0, 2)

                query_embed = self.position_embeddings.weight.unsqueeze(1)
                query_embed = query_embed.repeat(1, bs, 1)

                # query_embed_seq = self.position_embeddings_seq.weight.unsqueeze(1)
                # query_embed_seq = query_embed_seq.repeat(1, bs, 1)

                # prev_query_embed = self.previous_position_embeddings.weight.unsqueeze(1)
                # prev_query_embed = prev_query_embed.repeat(1, bs, 1)
                # prev_query_embed_encoder = prev_query_embed.permute(1, 0, 2)

                # x1_embed = prev_query_embed[:(self.pre_number)]
                # y1_embed = prev_query_embed[(self.pre_number):2*(self.pre_number)]
                # x2_embed = prev_query_embed[2*(self.pre_number):3*(self.pre_number)]
                # y2_embed = prev_query_embed[3*(self.pre_number):]

                # # input_seq = seqs_input.permute(1,0) # b * 30
                # input_seq = seqs_input
                # # print(input_seq.shape)
                # input_seq = input_seq.to(zx_feat.device).to(torch.int32)
                # input_seq = self.word_embeddings(input_seq)

                # seq_x1 = input_seq[:(self.pre_number),:,:]
                # seq_y1 = input_seq[(self.pre_number):2*(self.pre_number),:,:]
                # seq_x2 = input_seq[2*(self.pre_number):3*(self.pre_number),:,:]
                # seq_y2 = input_seq[3*(self.pre_number):,:,:]

                # print(seq_x1.shape,seq_y1.shape,seq_x2.shape,seq_y2.shape)
                # input_seq, z, x, prev_query_embed, z_pos, x_pos

                # _, x_feat_seq = self.plus_encoder(input_seq, z_feat,x_feat, prev_query_embed_encoder,None,None)

                # x_feat_seq = self.norm(x_feat_seq)

                # print(pos_z.shape, pos_x.shape)
                # print(self.pos_x_seq().shape, self.pos_z_seq().shape)

                # pos_x_seq = self.pos_x_seq().unsqueeze(0)
                # pos_x_seq = pos_x
                # pos_z_seq = self.pos_z_seq().unsqueeze(0)

                # decoder_feat_seq = self.decoder_seq(tgt ,x_feat_seq, z_feat, x_feat, pos_z_seq, pos_x_seq, None, query_embed_seq,
                #                                 tgt_mask= None)
                
                # decoder_feat_seq = self.decoder_seq(tgt ,x_feat_seq, z_feat, x_feat, pos_z, pos_x, identity, query_embed,
                #                                 tgt_mask= None)
                # decoder_feat = self.decoder(tgt, None, z_feat, x_feat, self.pos_embed_z, self.pos_embed_x, 
                #                             None, None, query_embed,
                #                                 tgt_mask= generate_mask_x1y1x2y2(self.pre_number+1).to(share_weight.device))

                decoder_feat = self.decoder(tgt, zx_feat, self.pos_embed_total,
                                            query_embed,
                                            tgt_mask= generate_mask_x1y1x2y2(self.pre_number+1).to(share_weight.device))

                out = torch.matmul(decoder_feat.transpose(0, 1), share_weight)
            
                out +=  self.output_bias.expand(out.shape[0], out.shape[1], out.shape[2])

                # outfeat = out.permute(1,0,2)

                # out_attention = torch.matmul(attention_feat.transpose(0, 1), share_weight)
            
                # out_attention +=  self.output_bias.expand(out.shape[0], out.shape[1], out.shape[2])
                # out_img_feat = out_img.permute(1,0,2)

                outfeat = out.permute(1,0,2)
                # out_img_feat = torch.zeros((outfeat.shape[0],outfeat.shape[1],outfeat.shape[2])).to(outfeat)

                out = out.softmax(-1)
                feat_for_hanning = out

                value, extra_seq = out.topk(dim=-1, k=1)[0], out.topk(dim=-1, k=1)[1]
                # seqs_input = torch.cat([seqs_input, extra_seq], dim=-1)
                seqs_output = extra_seq
                values = value
                
                # output = {'seqs': seqs_output, 'class': values, "state": "val/test"}
                # output = {'feat':outfeat,'seqs': seqs_output, 'class': values, "state": "val/test",
                #           'confidence':value}
                output = {'feat':outfeat,'seqs': seqs_output, 'class': values, "state": "val/test",
                          'confidence':value,'feat_for_hanning':feat_for_hanning}
                        # 'seq':decoder_feat_detach,'seq_emd':query_embed_detach}
            
            # out_dict['seq'],out_dict['x_feat']
        else:
            bs = seqs_input.shape[1]

            prev_query_embed = self.previous_position_embeddings.weight.unsqueeze(1)
            prev_query_embed = prev_query_embed.repeat(1, bs, 1)
            prev_query_embed_encoder = prev_query_embed.permute(1, 0, 2)

            x1_embed = prev_query_embed[:(self.pre_number)]
            y1_embed = prev_query_embed[(self.pre_number):2*(self.pre_number)]
            x2_embed = prev_query_embed[2*(self.pre_number):3*(self.pre_number)]
            y2_embed = prev_query_embed[3*(self.pre_number):]

            # input_seq = seqs_input.permute(1,0) # b * 30
            input_seq = seqs_input
            # print(input_seq.shape)
            input_seq = input_seq.to(share_weight.device).to(torch.int32)
            input_seq = self.word_embeddings(input_seq)

            seq_x1 = input_seq[:(self.pre_number),:,:]
            seq_y1 = input_seq[(self.pre_number):2*(self.pre_number),:,:]
            seq_x2 = input_seq[2*(self.pre_number):3*(self.pre_number),:,:]
            seq_y2 = input_seq[3*(self.pre_number):,:,:]

            init_feat = torch.from_numpy(seq).to(share_weight.device)
            query_embed =torch.from_numpy(seq_emd).to(share_weight.device)

            decoder_feat_seq = self.decoder_seq(seq_x1,seq_y1,seq_x2,seq_y2,
                                                x1_embed,y1_embed,x2_embed,y2_embed,
                                                init_feat ,None, None, None, None, None, None, None, query_embed,
                                                tgt_mask= generate_mask_x1y1x2y2(self.pre_number+1).to(share_weight.device),
                                                tgt_mask_onlyseq = None
                                                )
            out = torch.matmul(decoder_feat_seq.transpose(0, 1), share_weight)
            
            out +=  self.output_bias_seq.expand(out.shape[0], out.shape[1], out.shape[2])

            outfeat = out.permute(1,0,2)

            out = out.softmax(-1)

            value, extra_seq = out.topk(dim=-1, k=1)[0], out.topk(dim=-1, k=1)[1]
            # seqs_input = torch.cat([seqs_input, extra_seq], dim=-1)
            seqs_output = extra_seq
            values = value
            # print('feat',outfeat.shape)
            output = {'feat':outfeat,'seqs': seqs_output, 'class': values, "state": "val/test",
                    }
            
        return output

def build_pix_head(cfg, hidden_dim):
    stride = cfg.MODEL.BACKBONE.STRIDE

    if cfg.MODEL.HEAD.TYPE == "MLP":
        mlp_head = MLP(hidden_dim, hidden_dim, 4, 3)  # dim_in, dim_hidden, dim_out, 3 layers
        return mlp_head
    elif "CORNER" in cfg.MODEL.HEAD.TYPE:
        feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
        channel = getattr(cfg.MODEL, "NUM_CHANNELS", 256)
        print("head channel: %d" % channel)
        if cfg.MODEL.HEAD.TYPE == "CORNER":
            corner_head = Corner_Predictor(inplanes=cfg.MODEL.HIDDEN_DIM, channel=channel,
                                           feat_sz=feat_sz, stride=stride)
        else:
            raise ValueError()
        return corner_head
    elif cfg.MODEL.HEAD.TYPE == "CENTER":
        in_channel = hidden_dim
        out_channel = cfg.MODEL.HEAD.NUM_CHANNELS
        feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
        center_head = CenterPredictor(inplanes=in_channel, channel=out_channel,
                                      feat_sz=feat_sz, stride=stride)
        return center_head
    elif cfg.MODEL.HEAD.TYPE == "PIX":
        in_channel = hidden_dim
        feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
        feat_tz = int(cfg.DATA.TEMPLATE.SIZE / stride)
        decoder_layer = cfg.MODEL.DECODER_LAYER
        encoder_layer = cfg.MODEL.ENCODER_LAYER
        bins = cfg.MODEL.BINS
        num_heads = cfg.MODEL.NUM_HEADS
        mlp_ratio = cfg.MODEL.MLP_RATIO
        qkv_bias = cfg.MODEL.QKV_BIAS
        drop_rate = cfg.MODEL.DROP_RATE
        attn_drop = cfg.MODEL.ATTN_DROP
        drop_path = cfg.MODEL.DROP_PATH
        drop_path_allocator = DropPathAllocator(drop_path)
        range = cfg.MODEL.RANGE
        pre_number = cfg.DATA.SEARCH.NUMBER -1
        pix_head = Pix2Track(in_channel=in_channel, feat_sz=feat_sz, feat_tz=feat_tz, range=range,pre_number=pre_number,
                             stride=stride, encoder_layer=encoder_layer, decoder_layer=decoder_layer, bins=bins,
                             num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_rate=drop_rate,
                             attn_drop=attn_drop, drop_path=drop_path_allocator)
        return pix_head
    else:
        raise ValueError("HEAD TYPE %s is not supported." % cfg.MODEL.HEAD_TYPE)


