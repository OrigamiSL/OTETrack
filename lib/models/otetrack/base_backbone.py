from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import resize_pos_embed
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from lib.models.layers.patch_embed import PatchEmbed
from lib.models.otetrack.utils import combine_tokens, recover_tokens

class BaseBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        # for original ViT
        self.pos_embed = None
        self.img_size = [224, 224]
        self.patch_size = 16
        self.embed_dim = 384

        self.cat_mode = 'direct'

        self.pos_embed_z = None
        self.pos_embed_x = None

        self.template_segment_pos_embed = None
        self.search_segment_pos_embed = None

        self.return_inter = False
        self.return_stage = [2, 5, 8, 11]

        self.add_cls_token = False
        self.add_sep_seg = False

        self.x_patch_number_hw = 16
        self.z_patch_number_hw = 8

    def finetune_track(self, cfg, patch_start_index=1):

        search_size = to_2tuple(cfg.DATA.SEARCH.SIZE)
        template_size = to_2tuple(cfg.DATA.TEMPLATE.SIZE)
        new_patch_size = cfg.MODEL.BACKBONE.STRIDE

        self.cat_mode = cfg.MODEL.BACKBONE.CAT_MODE
        self.return_inter = cfg.MODEL.RETURN_INTER
        self.add_sep_seg = cfg.MODEL.BACKBONE.SEP_SEG

        # resize patch embedding
        if new_patch_size != self.patch_size:
            print('Inconsistent Patch Size With The Pretrained Weights, Interpolate The Weight!')
            old_patch_embed = {}
            for name, param in self.patch_embed.named_parameters():
                if 'weight' in name:
                    param = nn.functional.interpolate(param, size=(new_patch_size, new_patch_size),
                                                      mode='bicubic', align_corners=False)
                    param = nn.Parameter(param)
                old_patch_embed[name] = param
            self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=new_patch_size, in_chans=3,
                                          embed_dim=self.embed_dim,stride_size= 8)
            self.patch_embed.proj.bias = old_patch_embed['proj.bias']
            self.patch_embed.proj.weight = old_patch_embed['proj.weight']
        
        # self.patch_embed_next = PatchEmbed(img_size=self.img_size, patch_size=new_patch_size, in_chans=3,
        #                                   embed_dim=self.embed_dim,stride_size= 8)

        # print('self.patch_embed ',self.patch_embed.shape)
        # for patch embedding
        # print('self.pos_embed',self.pos_embed.shape)
        patch_pos_embed = self.pos_embed[:, patch_start_index:, :]
        patch_pos_embed = patch_pos_embed.transpose(1, 2)
        B, E, Q = patch_pos_embed.shape
        P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)
        # print('patch_pos_embed',patch_pos_embed.shape)
        # for search region
        H, W = search_size

        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.x_patch_number_hw = new_P_H
        search_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(2*new_P_H-1, 2*new_P_W-1), mode='bicubic',
                                                           align_corners=False)
        # search_patch_pos_embed = search_patch_pos_embed.flatten(2).transpose(1, 2)

        # search_patch_pos_embed_next = nn.functional.interpolate(patch_pos_embed, size=(new_P_H-1, new_P_W-1), mode='bicubic',
        #                                                    align_corners=False)
        # search_patch_pos_embed_next = search_patch_pos_embed_next.flatten(2).transpose(1, 2)
        # print(search_patch_pos_embed.shape)

        # for template region
        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.z_patch_number_hw = new_P_H
        template_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(2*new_P_H-1, 2*new_P_W-1), mode='bicubic',
                                                             align_corners=False)
        
        template_patch_pos_embed_1 = nn.functional.interpolate(patch_pos_embed, size=(2*new_P_H-1, 2*new_P_W-1), mode='bicubic',
                                                             align_corners=False)
        # template_patch_pos_embed = template_patch_pos_embed.flatten(2).transpose(1, 2)

        # template_patch_pos_embed_next = nn.functional.interpolate(patch_pos_embed, size=(new_P_H-1, new_P_W-1), mode='bicubic',
        #                                                      align_corners=False)
        # template_patch_pos_embed_next = template_patch_pos_embed_next.flatten(2).transpose(1, 2)
        # print(template_patch_pos_embed.shape)

        self.pos_embed_z = nn.Parameter(template_patch_pos_embed[:,:,0::2,0::2].flatten(2).transpose(1, 2))
        self.pos_embed_z_1 = nn.Parameter(template_patch_pos_embed_1[:,:,0::2,0::2].flatten(2).transpose(1, 2))
        self.pos_embed_x = nn.Parameter(search_patch_pos_embed[:,:,0::2,0::2].flatten(2).transpose(1, 2))

        self.pos_embed_z_next = nn.Parameter(template_patch_pos_embed[:,:,1::2,1::2].flatten(2).transpose(1, 2))
        self.pos_embed_z_next_1 = nn.Parameter(template_patch_pos_embed_1[:,:,1::2,1::2].flatten(2).transpose(1, 2))
        self.pos_embed_x_next = nn.Parameter(search_patch_pos_embed[:,:,1::2,1::2].flatten(2).transpose(1, 2))

        # for cls token (keep it but not used)
        if self.add_cls_token and patch_start_index > 0:
            cls_pos_embed = self.pos_embed[:, 0:1, :]
            self.cls_pos_embed = nn.Parameter(cls_pos_embed)

        # separate token and segment token
        if self.add_sep_seg:
            self.template_segment_pos_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.template_segment_pos_embed = trunc_normal_(self.template_segment_pos_embed, std=.02)
            self.search_segment_pos_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.search_segment_pos_embed = trunc_normal_(self.search_segment_pos_embed, std=.02)

        if self.return_inter:
            for i_layer in self.fpn_stage:
                if i_layer != 11:
                    norm_layer = partial(nn.LayerNorm, eps=1e-6)
                    layer = norm_layer(self.embed_dim)
                    layer_name = f'norm{i_layer}'
                    self.add_module(layer_name, layer)

    def forward_features(self, z, x, identity):
        # print(z.shape)
        B, H, W = x.shape[0], x.shape[2], x.shape[3]
        z1 = z[1]
        z = z[0]

        x_all = (2*self.x_patch_number_hw-1)
        z_all = (2*self.z_patch_number_hw-1)

        # x_next = self.patch_embed_next(x)
        # z_next = self.patch_embed_next(z)

        # x_next = x_next[:,1::2,:] #15*15
        # z_next = z_next[:,1::2,:] #7*7

        x = self.patch_embed(x) #31*31
        z = self.patch_embed(z) #15*15
        z1 = self.patch_embed(z1) 

        C = x.shape[2]

        x = x.contiguous().view(B,x_all,x_all,C)
        z = z.contiguous().view(B,z_all,z_all,C)
        z1 = z1.contiguous().view(B,z_all,z_all,C)

        x_next = x[:,1::2,1::2,:].contiguous().view(B,(self.x_patch_number_hw-1)**2,C) #15*15
        z_next = z[:,1::2,1::2,:].contiguous().view(B,(self.z_patch_number_hw-1)**2,C) #7*7
        z1_next = z1[:,1::2,1::2,:].contiguous().view(B,(self.z_patch_number_hw-1)**2,C)

        x = x[:,0::2,0::2,:].contiguous().view(B,(self.x_patch_number_hw)**2,C) #16*16
        z = z[:,0::2,0::2,:].contiguous().view(B,(self.z_patch_number_hw)**2,C) #8*8
        z1 = z1[:,0::2,0::2,:].contiguous().view(B,(self.z_patch_number_hw)**2,C) 

        s_x = x.shape[1]
        s_z = z.shape[1]

        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        z += self.pos_embed_z
        z1 += self.pos_embed_z_1
        x += self.pos_embed_x
        
        # z += identity[:, 0, :].repeat(B, self.pos_embed_z.shape[1], 1)
        # x += identity[:, 1, :].repeat(B, self.pos_embed_x.shape[1], 1)
        z_next += self.pos_embed_z_next
        z1_next += self.pos_embed_z_next_1
        x_next += self.pos_embed_x_next

        if self.add_sep_seg:
            x += self.search_segment_pos_embed
            z += self.template_segment_pos_embed

        # x = combine_tokens(z, x, mode=self.cat_mode)
        x = torch.cat([x,x_next,z,z1,z_next,z1_next],dim =1)  

        if self.add_cls_token:
            x = torch.cat([cls_tokens, x], dim=1)

        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x)

        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]
        #x = recover_tokens(x, lens_z, lens_x, mode=self.cat_mode)
        x = self.norm(x)

        # out_x = x[:,0:(self.x_patch_number_hw)**2,:]
        # out_z = x[:,(self.x_patch_number_hw)**2+(self.x_patch_number_hw-1)**2:(self.x_patch_number_hw)**2+(self.x_patch_number_hw-1)**2+(self.z_patch_number_hw)**2,:]
        # x = torch.concat([out_z,out_x],dim =1)
        # z, x
        # return self.norm(x)
        return x

    def forward(self, z, x, identity, **kwargs):
        """
        Joint feature extraction and relation modeling for the basic ViT backbone.
        Args:
            z (torch.Tensor): template feature, [B, C, H_z, W_z]
            x (torch.Tensor): search region feature, [B, C, H_x, W_x]

        Returns:
            x (torch.Tensor): merged template and search region feature, [B, L_z+L_x, C]
            attn : None
        """
        x = self.forward_features(z, x, identity)

        return x
