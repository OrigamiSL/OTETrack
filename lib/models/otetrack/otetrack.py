"""
Basic OSTrack model.
"""
import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from lib.models.layers.head import build_pix_head
from lib.models.otetrack.vit import vit_base_patch16_224, vit_large_patch16_224
from lib.utils.box_ops import box_xyxy_to_cxcywh


class OTETrack(nn.Module):
    """ This is the base class for OTETrack """

    def __init__(self, transformer, pix_head, hidden_dim):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
        """
        super().__init__()
        self.identity = torch.nn.Parameter(torch.zeros(1, 2, hidden_dim))
        self.identity = trunc_normal_(self.identity, std=.02)        
        
        self.backbone = transformer
        self.pix_head = pix_head
    
    
    def forward(self, template=None,
                search=None,
                seq_input=None,
                seq = None,seq_emd = None,flag = 0
                ):
        if flag == 0:
            x = self.backbone(z=template, x=search, identity=self.identity,)

            # Forward head
            feat_last = x
            if isinstance(x, list):
                feat_last = x[-1]
            
            pos_z = self.backbone.pos_embed_z
            pos_x = self.backbone.pos_embed_x
            
            out = self.forward_head(feat_last, pos_z, pos_x, self.identity, seq_input,flag = 0)

            out['backbone_feat'] = x
            return out
        else:
            out = self.forward_head(seq_input = seq_input,seq = seq, seq_emd = seq_emd,flag = 1)
            return out

    def forward_head(self, cat_feature=None, pos_z=None, pos_x=None, identity=None, seq_input=None,seq = None, seq_emd = None,flag = 0):
        
        if flag == 0:
            output_dict = self.pix_head(cat_feature, pos_z, pos_x, identity, seq_input,flag = 0)
            return output_dict
        
        else:
            output_dict = self.pix_head(seqs_input = seq_input,seq = seq, seq_emd = seq_emd,flag = 1)
            return output_dict


def build_otetrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    # set pre-trained path
    # pretrained_path = os.path.join('/home/lhg/work/fxy_ar1/AR_one_sequence/pretrained_models/')

    if cfg.MODEL.PRETRAIN_FILE and ('OTETrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224':
        print("i use vit_large")
        backbone = vit_large_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    pix_head = build_pix_head(cfg, hidden_dim)

    model = OTETrack(
        backbone,
        pix_head,
        hidden_dim,
    )
    if cfg.MODEL.PRETRAIN_PTH != "":
        load_from = cfg.MODEL.PRETRAIN_PTH
        checkpoint = torch.load(load_from, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + load_from)
    if 'OTETrack' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model
