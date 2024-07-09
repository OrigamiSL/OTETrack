from easydict import EasyDict as edict
import yaml

"""
Add default config for OSTrack.
"""
cfg = edict()

# MODEL
cfg.MODEL = edict()
cfg.MODEL.PRETRAIN_FILE = "mae_pretrain_vit_base.pth"
cfg.MODEL.PRETRAIN_PTH = ""
cfg.MODEL.EXTRA_MERGER = False

cfg.MODEL.RETURN_INTER = False
cfg.MODEL.RETURN_STAGES = [2, 5, 8, 11]

# MODEL.BACKBONE
cfg.MODEL.BACKBONE = edict()
cfg.MODEL.BACKBONE.TYPE = "vit_base_patch16_224"
cfg.MODEL.BACKBONE.STRIDE = 16
cfg.MODEL.BACKBONE.MID_PE = False
cfg.MODEL.BACKBONE.SEP_SEG = False
cfg.MODEL.BACKBONE.CAT_MODE = 'direct'
cfg.MODEL.BACKBONE.MERGE_LAYER = 0
cfg.MODEL.BACKBONE.ADD_CLS_TOKEN = False
cfg.MODEL.BACKBONE.CLS_TOKEN_USE_MODE = 'ignore'

cfg.MODEL.BACKBONE.CE_LOC = []
cfg.MODEL.BACKBONE.CE_KEEP_RATIO = []
cfg.MODEL.BACKBONE.CE_TEMPLATE_RANGE = 'ALL'  # choose between ALL, CTR_POINT, CTR_REC, GT_BOX

# MODEL.HEAD
cfg.MODEL.BINS = 400
cfg.MODEL.RANGE = 2
cfg.MODEL.ENCODER_LAYER = 3
# cfg.MODEL.NUM_HEADS = 12
cfg.MODEL.NUM_HEADS = 4
cfg.MODEL.MLP_RATIO = 4
cfg.MODEL.QKV_BIAS = True
cfg.MODEL.DROP_RATE = 0.1
cfg.MODEL.ATTN_DROP = 0.0
cfg.MODEL.DROP_PATH = 0.0
cfg.MODEL.DECODER_LAYER = 6
cfg.MODEL.HEAD = edict()
cfg.MODEL.HEAD.TYPE = "PIX"
cfg.MODEL.HEAD.NUM_CHANNELS = 1024

# TRAIN
cfg.TRAIN = edict()
cfg.TRAIN.LR = 0.0001
cfg.TRAIN.WEIGHT_DECAY = 0.0001
cfg.TRAIN.EPOCH = 500
cfg.TRAIN.LR_DROP_EPOCH = 400
cfg.TRAIN.BATCH_SIZE = 16
cfg.TRAIN.NUM_WORKER = 10
cfg.TRAIN.OPTIMIZER = "ADAMW"
cfg.TRAIN.BACKBONE_MULTIPLIER = 0.1
cfg.TRAIN.GIOU_WEIGHT = 2.0
cfg.TRAIN.L1_WEIGHT = 5.0
cfg.TRAIN.FREEZE_LAYERS = [0, ]
cfg.TRAIN.PRINT_INTERVAL = 50
cfg.TRAIN.VAL_EPOCH_INTERVAL = 20
cfg.TRAIN.GRAD_CLIP_NORM = 0.1
cfg.TRAIN.AMP = False

cfg.TRAIN.CE_START_EPOCH = 20  # candidate elimination start epoch
cfg.TRAIN.CE_WARM_EPOCH = 80  # candidate elimination warm up epoch
cfg.TRAIN.DROP_PATH_RATE = 0.1  # drop path rate for ViT backbone

# TRAIN.SCHEDULER
cfg.TRAIN.SCHEDULER = edict()
cfg.TRAIN.SCHEDULER.TYPE = "step"
cfg.TRAIN.SCHEDULER.DECAY_RATE = 0.1

# DATA
cfg.DATA = edict()
cfg.DATA.SAMPLER_MODE = "causal"  # sampling methods
cfg.DATA.MEAN = [0.485, 0.456, 0.406]
cfg.DATA.STD = [0.229, 0.224, 0.225]
cfg.DATA.MAX_SAMPLE_INTERVAL = 200
# DATA.TRAIN
cfg.DATA.TRAIN = edict()
cfg.DATA.TRAIN.DATASETS_NAME = ["LASOT", "GOT10K_vottrain"]
cfg.DATA.TRAIN.DATASETS_RATIO = [1, 1]
cfg.DATA.TRAIN.DATASETS_NAME_seq = ["LASOT", "GOT10K_vottrain"]
cfg.DATA.TRAIN.DATASETS_RATIO_seq = [1, 1]
cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 60000
# DATA.VAL
cfg.DATA.VAL = edict()
cfg.DATA.VAL.DATASETS_NAME = ["GOT10K_votval"]
cfg.DATA.VAL.DATASETS_RATIO = [1]
cfg.DATA.VAL.SAMPLE_PER_EPOCH = 10000
# DATA.SEARCH
cfg.DATA.SEARCH = edict()
cfg.DATA.SEARCH.SIZE = 256
cfg.DATA.SEARCH.FACTOR = 5.0
cfg.DATA.SEARCH.CENTER_JITTER = 4.5
cfg.DATA.SEARCH.SCALE_JITTER = 0.5
cfg.DATA.SEARCH.NUMBER = 1
cfg.DATA.SEARCH.USE_PREDICT= 7
# DATA.TEMPLATE
cfg.DATA.TEMPLATE = edict()
cfg.DATA.TEMPLATE.NUMBER = 1
cfg.DATA.TEMPLATE.SIZE = 128
cfg.DATA.TEMPLATE.FACTOR = 2.0
cfg.DATA.TEMPLATE.CENTER_JITTER = 0
cfg.DATA.TEMPLATE.SCALE_JITTER = 0

# TEST
cfg.TEST = edict()
cfg.TEST.TEMPLATE_FACTOR = 2.0
cfg.TEST.TEMPLATE_SIZE = 128
cfg.TEST.SEARCH_FACTOR = 5.0
cfg.TEST.SEARCH_SIZE = 256
cfg.TEST.EPOCH = 500
cfg.TEST.TEMPLATE_NUMBER = 2

# dataset_name = ['lasot','got10k_test','trackingnet','lasot_extension_subset','tnl2k','nfs','uav']
cfg.TEST.UPDATE_INTERVALS = edict()
cfg.TEST.UPDATE_INTERVALS.DEFAULT = 25
cfg.TEST.UPDATE_INTERVALS.LASOT = 25
cfg.TEST.UPDATE_INTERVALS.GOT10K_TEST = 25
cfg.TEST.UPDATE_INTERVALS.TRACKINGNET = 25
cfg.TEST.UPDATE_INTERVALS.LASOT_EXTENSION_SUBSET = 200
cfg.TEST.UPDATE_INTERVALS.UAV = 100

cfg.TEST.UPDATE_THRESHOLD = edict()
cfg.TEST.UPDATE_THRESHOLD.DEFAULT = 0.6
cfg.TEST.UPDATE_THRESHOLD.LASOT = 0.6
cfg.TEST.UPDATE_THRESHOLD.GOT10K_TEST = 0.65
cfg.TEST.UPDATE_THRESHOLD.TRACKINGNET = 0.7
cfg.TEST.UPDATE_THRESHOLD.LASOT_EXTENSION_SUBSET = 1.2
cfg.TEST.UPDATE_THRESHOLD.UAV = 0.6

cfg.TEST.SMOOTH_THRESHOLD= edict()
cfg.TEST.SMOOTH_THRESHOLD.DEFAULT = 0.6
cfg.TEST.SMOOTH_THRESHOLD.LASOT = 0.8
cfg.TEST.SMOOTH_THRESHOLD.GOT10K_TEST = 0.4
cfg.TEST.SMOOTH_THRESHOLD.TRACKINGNET = 4.0
cfg.TEST.SMOOTH_THRESHOLD.LASOT_EXTENSION_SUBSET = 1.3
cfg.TEST.SMOOTH_THRESHOLD.UAV = 1.4

cfg.TEST.HANNING_SIZE = edict()
cfg.TEST.HANNING_SIZE.DEFAULT = 6000
cfg.TEST.HANNING_SIZE.LASOT = 6000
cfg.TEST.HANNING_SIZE.GOT10K_TEST = 5500
cfg.TEST.HANNING_SIZE.TRACKINGNET = 6000
cfg.TEST.HANNING_SIZE.LASOT_EXTENSION_SUBSET = 2500
cfg.TEST.HANNING_SIZE.UAV = 1800

cfg.TEST.STD_WEIGHT= edict()
cfg.TEST.STD_WEIGHT.DEFAULT = 1
cfg.TEST.STD_WEIGHT.LASOT_EXTENSION_SUBSET = 1.1
cfg.TEST.STD_WEIGHT.UAV = 2

cfg.TEST.ALPHA = 0.7

cfg.TEST.BETA= edict()
cfg.TEST.BETA.DEFAULT = 0.8
cfg.TEST.BETA.LASOT_EXTENSION_SUBSET = 0.9
def _edict2dict(dest_dict, src_edict):
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for k, v in src_edict.items():
            if not isinstance(v, edict):
                dest_dict[k] = v
            else:
                dest_dict[k] = {}
                _edict2dict(dest_dict[k], v)
    else:
        return

def gen_config(config_file):
    cfg_dict = {}
    _edict2dict(cfg_dict, cfg)
    with open(config_file, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)

def _update_config(base_cfg, exp_cfg):
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
        for k, v in exp_cfg.items():
            if k in base_cfg:
                if not isinstance(v, dict):
                    base_cfg[k] = v
                else:
                    _update_config(base_cfg[k], v)
            else:
                raise ValueError("{} not exist in config.py".format(k))
    else:
        return


def update_config_from_file(filename, base_cfg=None):
    exp_config = None
    with open(filename) as f:
        exp_config = edict(yaml.safe_load(f))
        if base_cfg is not None:
            _update_config(base_cfg, exp_config)
        else:
            _update_config(cfg, exp_config)
