import math

from lib.models.otetrack import build_otetrack
from lib.test.tracker.basetracker import BaseTracker
import torch
import glob

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target, transform_image_to_crop
# for debug
import cv2
import os
import numpy as np
from scipy.optimize import leastsq
from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond
import random

from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import Holt


def get_hanning(std,box_out_last,feat_for_hanning,bins,magic_num,hanning_size):
    all_lenth = 4 * bins # 8000
    half_lenth = 2 * bins
    padding_size = int((all_lenth - hanning_size) / 2)
    # print('padding_size',padding_size)
    
    init_hanning = np.hanning(hanning_size+2)
    # print(len(init_hanning))
    init_hanning = list(init_hanning)
    padding = [ 0 for i in range(padding_size)]
    all_list = padding + init_hanning + padding
    init_hanning = all_list

    # init_hanning = np.hanning(all_lenth+1)
    # init_hanning = list(init_hanning)

    x1_pre, y1_pre, x2_pre, y2_pre = int(np.around(box_out_last[0].item())),int(np.around(box_out_last[1].item())),\
        int(np.around(box_out_last[2].item())),int(np.around(box_out_last[3].item()))
    # print('box',box_out_last)

    std[0],std[1],std[2],std[3] = int(np.ceil(std[0].item())),int(np.ceil(std[1].item())),\
        int(np.ceil(std[2].item())),int(np.ceil(std[3].item()))
    # print('std',std)
    # print('label',x1_pre, y1_pre, x2_pre, y2_pre) 
    
    left_x1, right_x1 = x1_pre , half_lenth +2 - x1_pre
    left_y1, right_y1 = y1_pre , half_lenth +2 - y1_pre
    left_x2, right_x2 = x2_pre , half_lenth +2 - x2_pre
    left_y2, right_y2 = y2_pre , half_lenth +2 - y2_pre

    def get_hanning(init_hanning,half_lenth,left,right,std):
        if left >= std:
            left_part = init_hanning[half_lenth-(left-std):half_lenth]
            middle_pad_left = [ 1 for i in range(std)]
        else:
            left_part = []
            middle_pad_left = [ 1 for i in range(left)]

        if right >= std:
            right_part = init_hanning[half_lenth:half_lenth+(right-std)]
            middle_pad_right = [ 1 for i in range(std)]
        else:
            right_part = []
            middle_pad_right = [ 1 for i in range(right)]
        # 0 0,4002,0
        # print(len(left_part),len(middle_pad_left),len(middle_pad_right),len(right_part))
        return np.array(left_part + middle_pad_left +middle_pad_right + right_part)

    x1_hanning = get_hanning(init_hanning,half_lenth,left_x1,right_x1,std[0])
    y1_hanning = get_hanning(init_hanning,half_lenth,left_y1,right_y1,std[1])
    # print('y1',left_y1,right_y1,std[1])
    x2_hanning = get_hanning(init_hanning,half_lenth,left_x2,right_x2,std[2])
    y2_hanning = get_hanning(init_hanning,half_lenth,left_y2,right_y2,std[3])
    
    output = torch.zeros_like(feat_for_hanning)
    output[:,0,:] = feat_for_hanning[:,0,:] * torch.tensor(x1_hanning).unsqueeze(0).cuda()
    output[:,1,:] = feat_for_hanning[:,1,:] * torch.tensor(y1_hanning).unsqueeze(0).cuda()
    output[:,2,:] = feat_for_hanning[:,2,:] * torch.tensor(x2_hanning).unsqueeze(0).cuda()
    output[:,3,:] = feat_for_hanning[:,3,:] * torch.tensor(y2_hanning).unsqueeze(0).cuda()
    # print('output',output.shape)

    return output
def SimpleSmoothing(seq, multi_weight):
    seq_len = len(seq)

    final = 0
    for i in range(seq_len):
        if i == 0:
            x1 = ((1-multi_weight)**(seq_len-1-i))*seq[i]
        else:
            x1 = multi_weight*((1-multi_weight)**(seq_len-1-i))*seq[i]
        final += x1
    
    return final
def DoubleSmoothing(series, alpha, beta):
    result = [series[0]]
    for n in range(1, len(series)+1):
        if n == 1:
            level, trend = series[0], series[1] - series[0]
        if n >= len(series): # forecasting
            value = result[-1]
        else:
            value = series[n]
        last_level, level = level, alpha*value + (1-alpha)*(level+trend)
        trend = beta*(level-last_level) + (1-beta)*trend
        result.append(level+trend)
    
    result = result[-1]
    return result
class OTETrack(BaseTracker):
    def __init__(self, params, dataset_name, test_checkpoint,net_for_test,update_intervals,update_threshold,hanning_size,\
                 pre_seq_number,std_weight, smooth_type, alpha, beta, double_dayu, smooth_thre):
        super(OTETrack, self).__init__(params)
        if net_for_test == None:
            print('build new network for test')
            network = build_otetrack(params.cfg, training=False)
        else:
            print('use trained network for test')
            network = net_for_test
        
        if test_checkpoint == None:
            print('dont use trained checkpoint')
        else:
            print('load_checkpoint_path',test_checkpoint)
            network.load_state_dict(torch.load(test_checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.bins = self.cfg.MODEL.BINS
        # self.use_frame_number = self.cfg.DATA.SEARCH.NUMBER
        self.use_frame_number = self.cfg.DATA.SEARCH.NUMBER
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None
        self.range = self.cfg.MODEL.RANGE
        self.num_template = self.cfg.TEST.TEMPLATE_NUMBER
        print('template number',self.num_template)

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        self.debug = False
        self.use_visdom = False
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
        # self.save_dir = ''
        # if not os.path.exists(self.save_dir):
        #     os.makedirs(self.save_dir)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.dataset_name = dataset_name

        self.z_dict1 = {}
        DATASET_NAME = dataset_name.upper()

        if hasattr(self.cfg.TEST.UPDATE_INTERVALS, DATASET_NAME):
            self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS[DATASET_NAME]
        else:
            print('use_default_inter')
            self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS.DEFAULT
        print("Update interval is: ", self.update_intervals)
        
        if hasattr(self.cfg.TEST.UPDATE_THRESHOLD, DATASET_NAME):
            self.update_threshold = self.cfg.TEST.UPDATE_THRESHOLD[DATASET_NAME]
        else:
            print('use_default_thre')
            self.update_threshold = self.cfg.TEST.UPDATE_THRESHOLD.DEFAULT
        print("Update threshold is: ", self.update_threshold)   

        self.current_frame = 1
        self.multi_weight = 0.9
        print('self.multi_weight',self.multi_weight)
        # self.hanning_size = 6000
        if hasattr(self.cfg.TEST.HANNING_SIZE,DATASET_NAME):
            self.hanning_size = self.cfg.TEST.HANNING_SIZE[DATASET_NAME]
        else:
            print('use_default_smooth_thre')
            self.hanning_size = self.cfg.TEST.HANNING_SIZE.DEFAULT
        # if hanning_size == None:
        #     self.hanning_size = 6000
        # else:
        #     self.hanning_size = int(hanning_size)
        print('self.hanning_size',self.hanning_size)

        if pre_seq_number  == None:
            if dataset_name == 'uav':
                self.pre_seq_number = 5
            else:
                self.pre_seq_number = self.cfg.DATA.SEARCH.USE_PREDICT # 7
        else:
            self.pre_seq_number = int(pre_seq_number)
        print('self.pre_seq_number',self.pre_seq_number)

        # if std_weight == None:
        #     if dataset_name == 'uav':
        #         self.std_weight = 2
        #     else:
        #         self.std_weight = self.cfg.TEST.STD_WEIGHT.DEFAULT
        if hasattr(self.cfg.TEST.STD_WEIGHT,DATASET_NAME):
            self.std_weight = self.cfg.TEST.STD_WEIGHT[DATASET_NAME]
        else:
            print('use_default_STD_WEIGHT')
            self.std_weight = self.cfg.TEST.STD_WEIGHT.DEFAULT
        # else:
        #     self.std_weight = float(std_weight)

        print('self.std_weight',self.std_weight)
        
        if alpha != None:
            self.alpha = float(alpha)
        else:
            self.alpha = self.cfg.TEST.ALPHA
        print('alpha',self.alpha)

        if hasattr(self.cfg.TEST.BETA,DATASET_NAME):
            self.beta = self.cfg.TEST.BETA[DATASET_NAME]
        else:
            print('use_default_beta')
            self.beta = self.cfg.TEST.BETA.DEFAULT
        print('beta',self.beta)


        if hasattr(self.cfg.TEST.SMOOTH_THRESHOLD, DATASET_NAME):
            self.smooth_thre = self.cfg.TEST.SMOOTH_THRESHOLD[DATASET_NAME]
        else:
            print('use_default_smooth_thre')
            self.smooth_thre = self.cfg.TEST.SMOOTH_THRESHOLD.DEFAULT
        # self.smooth_thre = float(smooth_thre)
        print('smooth_thre', self.smooth_thre)
       
    def initialize(self, image, info: dict):
        # forward the template once

        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)#output_sz=self.params.template_size
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)

        with torch.no_grad():
            self.z_dict1 = template
            self.template_list = [template] * self.num_template

        self.box_mask_z = None

        # save states
        self.state = info['init_bbox']
        
        init_box = info['init_bbox']

        self.store_result = [init_box]
        for i in range(self.pre_seq_number - 1):
            self.store_result.append(init_box)

        # self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, current_frame,info: dict = None):
        magic_num = (self.range - 1) * 0.5
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        box_center = torch.zeros(4*len(self.store_result))


        with torch.no_grad():
            x_dict = search
            template_list = []
            for i in self.template_list:
                template_list.append(i.tensors)
            
            out_dict = self.network.forward(
                template = template_list , search=x_dict.tensors,seq_input=box_center)
        
        feat_for_hanning = out_dict['feat_for_hanning']
        conf_score = out_dict['confidence'].sum().item() * 10
        
        if conf_score > self.smooth_thre:
            self.smooth_type = 'double'
            h_z = self.hanning_size
            # self.hanning_size = 6000
        else:
            self.smooth_type = 'single'
            if self.dataset_name == 'lasot_extension_subset':
                h_z = 3500
            else:
                h_z = 6000
               
        # use_predict
        previous_seq = []
        if self.current_frame <= self.pre_seq_number :
            previous_seq = self.store_result[-self.current_frame:]
           
            assert len(previous_seq) <= self.pre_seq_number 
        else:
            previous_seq = self.store_result
        
        previous = np.zeros(4*len(previous_seq))
        for e in range(len(previous_seq)):
            y = previous_seq[e]
            x_1 = y[0] 
            y_1 = y[1] 
            x_2 = y[0] + y[2] 
            y_2 = y[1] + y[3] 

            previous[0+4*e] = x_1
            previous[1+4*e] = y_1
            previous[2+4*e] = x_2 - x_1
            previous[3+4*e] = y_2 - y_1
           
        previous_label_x = np.zeros(len(previous_seq))
        previous_label_y = np.zeros(len(previous_seq))

        weighted_label = np.zeros((4))
        all_weight = np.sum([self.multi_weight**(i) for i in range(len(previous_seq))])
        # print('all_weight',all_weight)
        if len(previous_seq) > 1:
            for i in range(len(previous_seq)):
                box_out_i = transform_image_to_crop(torch.Tensor(previous[0+4*i:4+4*i]), torch.Tensor(self.state),None,
                                                            resize_factor,
                                                            torch.Tensor([self.cfg.TEST.SEARCH_SIZE, self.cfg.TEST.SEARCH_SIZE]),
                                                            normalize=True)
                box_out_i[2] = box_out_i[2] + box_out_i[0]
                box_out_i[3] = box_out_i[3] + box_out_i[1]
                box_out_i = box_out_i.clamp(min=-0.5, max=1.5)
                box_out_i = (box_out_i + 0.5) * (self.bins - 1)
               
                previous_label_x[i] = (box_out_i[0] + box_out_i[2]) / 2
                previous_label_y[i] = (box_out_i[3] + box_out_i[1]) / 2
        
            x1_std = np.std(previous_label_x)
            y1_std = np.std(previous_label_y)
            x2_std = np.std(previous_label_x)
            y2_std = np.std(previous_label_y)
        else:
            x1_std ,y1_std, x2_std, y2_std = torch.Tensor([5]),torch.Tensor([5]),torch.Tensor([5]),torch.Tensor([5])

        std = [x1_std, y1_std, x2_std, y2_std]
        # print(std)
        if len(previous_seq) > 2:
            
            previous_x = np.zeros(len(previous_seq))
            previous_y = np.zeros(len(previous_seq))

            for i in range(len(previous_seq)):
                y = previous_seq[i]
                xc = y[0] + y[2]/2
                yc = y[1] + y[3]/2
                previous_x[i] = xc
                previous_y[i] = yc

            if self.smooth_type == 'single':
                res_x = SimpleSmoothing(previous_x,self.multi_weight)
                res_y = SimpleSmoothing(previous_y,self.multi_weight)

            elif self.smooth_type == 'double':
                fit_x = Holt(previous_x).fit(smoothing_level= self.alpha, smoothing_trend= self.beta, optimized= False)
                fit_y = Holt(previous_y).fit(smoothing_level= self.alpha, smoothing_trend= self.beta, optimized= False)
                res_x = fit_x.forecast(1)[0]
                res_y = fit_y.forecast(1)[0]
            else:
                print('NO smooth_type')
            
            weighted_label[0] = res_x - self.state[2]/2
            weighted_label[1] = res_y - self.state[3]/2
            weighted_label[2] = self.state[2]
            weighted_label[3] = self.state[3]
        
        else:
            weighted_label = self.state
        
        box_out_last = transform_image_to_crop(torch.Tensor(weighted_label), torch.Tensor(self.state),None,
                                                        resize_factor,
                                                        torch.Tensor([self.cfg.TEST.SEARCH_SIZE, self.cfg.TEST.SEARCH_SIZE]),
                                                        normalize=True)
        box_out_last[2] = box_out_last[2] + box_out_last[0]
        box_out_last[3] = box_out_last[3] + box_out_last[1]
        box_out_last = box_out_last.clamp(min=-0.5, max=1.5)
        box_out_last = (box_out_last + 0.5) * (self.bins - 1)
       
        input_std = []
        for xx_std in std:
            input_std.append(self.std_weight*xx_std)
        
        hanning_feat = get_hanning(input_std,box_out_last,feat_for_hanning,self.bins,magic_num,h_z)
        
        value, extra_seq = hanning_feat.topk(dim=-1, k=1)[0], hanning_feat.topk(dim=-1, k=1)[1]
        
        pred_boxes = extra_seq[:, 0:4] / (self.bins - 1) - magic_num

        pred_boxes = pred_boxes.view(-1, 4).mean(dim=0)
        pred_new = pred_boxes
        pred_new[2] = pred_boxes[2] - pred_boxes[0]
        pred_new[3] = pred_boxes[3] - pred_boxes[1]
        pred_new[0] = pred_boxes[0] + pred_boxes[2]/2
        pred_new[1] = pred_boxes[1] + pred_boxes[3]/2
        pred_boxes = (pred_new * self.params.search_size / resize_factor).tolist()
        self.state = clip_box(self.map_box_back(pred_boxes, resize_factor), H, W, margin=10)

        # update the template
        if self.num_template > 1:
            conf_score = out_dict['confidence'].sum().item() * 10 # the confidence score
            if (self.frame_id % self.update_intervals == 0) and (conf_score > self.update_threshold):
                z_patch_arr, _, z_amask_arr= sample_target(image, self.state, self.params.template_factor,
                                               output_sz=self.params.template_size)
                template = self.preprocessor.process(z_patch_arr,z_amask_arr)
                self.template_list.append(template)
                if len(self.template_list) > self.num_template:
                    self.template_list.pop(1)
        
        current_bbox = []
        current_bbox = self.state
        new_result = []
        for i in range(self.pre_seq_number):
            if i != self.pre_seq_number - 1:
                new_result = self.store_result[i + 1]
                self.store_result[i] = new_result
            else:
                self.store_result[i] =  current_bbox
        
        self.current_frame += 1

        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)


            else:
                self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')

                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor

        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        #cx_real = cx + cx_prev
        #cy_real = cy + cy_prev
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return OTETrack
