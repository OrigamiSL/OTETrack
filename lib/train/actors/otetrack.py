from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
import math
import numpy as np
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate
import lib.train.data.bounding_box_utils as bbutils

def fp16_clamp(x, min=None, max=None):
    if not x.is_cuda and x.dtype == torch.float16:
        # clamp for cpu float16, tensor fp16 has no clamp implementation
        return x.float().clamp(min, max).half()

    return x.clamp(min, max)
    
def generate_sa_simdr(joints):
    '''
    :param joints:  [num_joints, 3]
    :param joints_vis: [num_joints, 3]
    :return: target, target_weight(1: visible, 0: invisible)
    '''
    num_joints = 48
    image_size = [256, 256]
    simdr_split_ratio = 1.5625
    sigma = 6

    target_x1 = np.zeros((num_joints,
                              int(image_size[0] * simdr_split_ratio)),
                             dtype=np.float32)
    target_y1 = np.zeros((num_joints,
                              int(image_size[1] * simdr_split_ratio)),
                             dtype=np.float32)
    target_x2 = np.zeros((num_joints,
                              int(image_size[0] * simdr_split_ratio)),
                             dtype=np.float32)
    target_y2 = np.zeros((num_joints,
                              int(image_size[1] * simdr_split_ratio)),
                             dtype=np.float32)
    zero_4_begin = np.zeros((num_joints, 1), dtype=np.float32)

    tmp_size = sigma * 3

    for joint_id in range(num_joints):

        mu_x1 = joints[joint_id][0]
        mu_y1 = joints[joint_id][1]
        mu_x2 = joints[joint_id][2]
        mu_y2 = joints[joint_id][3]

        x1 = np.arange(0, int(image_size[0] * simdr_split_ratio), 1, np.float32)
        y1 = np.arange(0, int(image_size[1] * simdr_split_ratio), 1, np.float32)
        x2 = np.arange(0, int(image_size[0] * simdr_split_ratio), 1, np.float32)
        y2 = np.arange(0, int(image_size[1] * simdr_split_ratio), 1, np.float32)

        target_x1[joint_id] = (np.exp(- ((x1 - mu_x1) ** 2) / (2 * sigma ** 2))) / (
                        sigma * np.sqrt(np.pi * 2))
        target_y1[joint_id] = (np.exp(- ((y1 - mu_y1) ** 2) / (2 * sigma ** 2))) / (
                        sigma * np.sqrt(np.pi * 2))
        target_x2[joint_id] = (np.exp(- ((x2 - mu_x2) ** 2) / (2 * sigma ** 2))) / (
                        sigma * np.sqrt(np.pi * 2))
        target_y2[joint_id] = (np.exp(- ((y2 - mu_y2) ** 2) / (2 * sigma ** 2))) / (
                        sigma * np.sqrt(np.pi * 2))
    return target_x1, target_y1, target_x2, target_y2

# angle cost
def SIoU_loss(test1, test2, theta=4):
    eps = 1e-7
    cx_pred = (test1[:, 0] + test1[:, 2]) / 2
    cy_pred = (test1[:, 1] + test1[:, 3]) / 2
    cx_gt = (test2[:, 0] + test2[:, 2]) / 2
    cy_gt = (test2[:, 1] + test2[:, 3]) / 2

    dist = ((cx_pred - cx_gt)**2 + (cy_pred - cy_gt)**2) ** 0.5
    ch = torch.max(cy_gt, cy_pred) - torch.min(cy_gt, cy_pred)
    x = ch / (dist + eps)

    angle = 1 - 2*torch.sin(torch.arcsin(x)-torch.pi/4)**2
    # distance cost
    xmin = torch.min(test1[:, 0], test2[:, 0])
    xmax = torch.max(test1[:, 2], test2[:, 2])
    ymin = torch.min(test1[:, 1], test2[:, 1])
    ymax = torch.max(test1[:, 3], test2[:, 3])
    cw = xmax - xmin
    ch = ymax - ymin
    px = ((cx_gt - cx_pred) / (cw+eps))**2
    py = ((cy_gt - cy_pred) / (ch+eps))**2
    gama = 2 - angle
    dis = (1 - torch.exp(-1 * gama * px)) + (1 - torch.exp(-1 * gama * py))

    #shape cost
    w_pred = test1[:, 2] - test1[:, 0]
    h_pred = test1[:, 3] - test1[:, 1]
    w_gt = test2[:, 2] - test2[:, 0]
    h_gt = test2[:, 3] - test2[:, 1]
    ww = torch.abs(w_pred - w_gt) / (torch.max(w_pred, w_gt) + eps)
    wh = torch.abs(h_gt - h_pred) / (torch.max(h_gt, h_pred) + eps)
    omega = (1 - torch.exp(-1 * wh)) ** theta + (1 - torch.exp(-1 * ww)) ** theta

    #IoU loss
    lt = torch.max(test1[..., :2], test2[..., :2])  # [B, rows, 2]
    rb = torch.min(test1[..., 2:], test2[..., 2:])  # [B, rows, 2]

    wh = fp16_clamp(rb - lt, min=0)
    overlap = wh[..., 0] * wh[..., 1]
    area1 = (test1[..., 2] - test1[..., 0]) * (
            test1[..., 3] - test1[..., 1])
    area2 = (test2[..., 2] - test2[..., 0]) * (
            test2[..., 3] - test2[..., 1])
    iou = overlap / (area1 + area2 - overlap)

    SIoU = 1 - iou + (omega + dis) / 2
    return SIoU, iou

def SIoU_loss_center(test1, test2, theta=4):
    # input:x_center,y_center,w,h
    eps = 1e-7
    cx_pred = test1[:, 0] 
    cy_pred = test1[:, 1] 
    cx_gt = test2[:, 0]
    cy_gt = test2[:, 1] 

    dist = ((cx_pred - cx_gt)**2 + (cy_pred - cy_gt)**2) ** 0.5
    ch = torch.max(cy_gt, cy_pred) - torch.min(cy_gt, cy_pred)
    x = ch / (dist + eps)

    angle = 1 - 2*torch.sin(torch.arcsin(x)-torch.pi/4)**2
    # distance cost
    xmin = torch.min(test1[:, 0]-test1[:, 2]/2, test2[:, 0]-test2[:, 2]/2)
    xmax = torch.max(test1[:, 0]+test1[:, 2]/2, test2[:, 0]+test2[:, 2]/2)
    ymin = torch.min(test1[:, 1]-test1[:, 3]/2, test2[:, 1]-test2[:, 3]/2)
    ymax = torch.max(test1[:, 1]+test1[:, 3]/2, test2[:, 1]+test2[:, 3]/2)
    cw = xmax - xmin
    ch = ymax - ymin
    px = ((cx_gt - cx_pred) / (cw+eps))**2
    py = ((cy_gt - cy_pred) / (ch+eps))**2
    gama = 2 - angle
    dis = (1 - torch.exp(-1 * gama * px)) + (1 - torch.exp(-1 * gama * py))

    #shape cost
    w_pred = test1[:, 2] 
    h_pred = test1[:, 3] 
    w_gt = test2[:, 2] 
    h_gt = test2[:, 3] 
    ww = torch.abs(w_pred - w_gt) / (torch.max(w_pred, w_gt) + eps)
    wh = torch.abs(h_gt - h_pred) / (torch.max(h_gt, h_pred) + eps)
    omega = (1 - torch.exp(-1 * wh)) ** theta + (1 - torch.exp(-1 * ww)) ** theta

    #IoU loss
    test1 = bbutils.batch_center2corner(test1) 
    test2 = bbutils.batch_center2corner(test2) 
    lt = torch.max(test1[..., :2], test2[..., :2])  # [B, rows, 2]
    rb = torch.min(test1[..., 2:], test2[..., 2:])  # [B, rows, 2]

    wh = fp16_clamp(rb - lt, min=0)
    overlap = wh[..., 0] * wh[..., 1]
    area1 = (test1[..., 2] - test1[..., 0]) * (
            test1[..., 3] - test1[..., 1])
    area2 = (test2[..., 2] - test2[..., 0]) * (
            test2[..., 3] - test2[..., 1])
    iou = overlap / (area1 + area2 - overlap)

    SIoU = 1 - iou + (omega + dis) / 2
    return SIoU, iou

def ciou(pred, target, eps=1e-7):
    # overlap
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]

    # union
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = ap + ag - overlap + eps

    # IoU
    ious = overlap / union

    # enclose area
    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)

    cw = enclose_wh[:, 0]
    ch = enclose_wh[:, 1]

    c2 = cw**2 + ch**2 + eps

    b1_x1, b1_y1 = pred[:, 0], pred[:, 1]
    b1_x2, b1_y2 = pred[:, 2], pred[:, 3]
    b2_x1, b2_y1 = target[:, 0], target[:, 1]
    b2_x2, b2_y2 = target[:, 2], target[:, 3]

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    left = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2))**2 / 4
    right = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2))**2 / 4
    rho2 = left + right

    factor = 4 / math.pi**2
    v = factor * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)

    # CIoU
    cious = ious - (rho2 / c2 + v**2 / (1 - ious + v))
    return cious, ious

def Attention_IOU_loss(pred, true):
    #b * 256
    batch_size = pred.shape[0]
    iou_all = torch.zeros(1).to(pred.device)
    
    for i in range(batch_size):
        pred_score = pred[i,:]
        true_map = true[i,:]
        target_patch_number = int(torch.sum(true_map).item())
        # print(target_patch_number)
        if (target_patch_number ==0):
            print('true_map_error')
            continue
        pred_index = pred_score.topk(k = target_patch_number,dim =0)[1].to('cpu')
        pred_label = torch.zeros((pred_score.shape[0])).to(pred_index.device)
        # print('pred_index',pred_index)
        
        pred_label.scatter_(0,torch.LongTensor(pred_index),1) 

        zero_label = torch.zeros((pred_score.shape[0])).to(pred_score.device)
        sum_label= true_map  +  pred_label.to(pred_score.device)

        # if(i ==0):
        #     print('pred_label')
        #     print(pred_label) 
        #     print('true_map')
        #     print(true_map)     
        # print('sum_label')
        # print(sum_label)
        intersection = torch.where(sum_label==1,sum_label,zero_label)
        intersection = torch.sum(intersection)
        # print('intersection',intersection)
        union = torch.where(sum_label==2,sum_label,zero_label)
        union = torch.sum(union) / 2
        # print('union',union)
        iou = union / (union + intersection)
        iou_all += iou
    # print('iou_all',iou_all)
    # exit(-1)
    iou_mean = iou_all / batch_size
    return 1 - iou_mean

def Compute_Contra_loss(predx, true):
    #predx: b * 256 * 768
    #predz: b * 64 * 768
    #true: b*256
    batch_size = predx.shape[0]
    total_patch_number = true.shape[1]
    patches_numbers = []
    for i in range(batch_size):
        pred_score = predx[i,:,:]
        # template_map = predz[i,:,:]
        true_map = true[i,:]
        background_map = 1 - true[i,:]

        target_patch_number = int(torch.sum(true_map).item())
        # print('1',target_patch_number)
        if (target_patch_number ==0):
            print('true_map_error')
            continue

        patches_numbers.append(target_patch_number)
        target_index = torch.nonzero(true_map).squeeze(1)
        background_index = torch.nonzero(background_map).squeeze(1)
        # choose_patch = np.random.choice([ii for ii in range(total_patch_number-target_patch_number)], target_patch_number)
        # background_index = background_index[choose_patch]
        # print(background_index.shape)
        # exit(-1)

        target = torch.index_select(pred_score,0,target_index)
        background = torch.index_select(pred_score,0,background_index)

        target_background = torch.cat([target,background],dim = 0).unsqueeze(0)
        if i == 0:
            input_tensor = target_background
            # target_tensor = get_target
        else:
            input_tensor = torch.cat([input_tensor,target_background],dim = 0) 
            # target_tensor = torch.cat([target_tensor,get_target],dim = 0)  

    # total_tensor = torch.cat([input_tensor,predz],dim = 1)
    B, P, D = input_tensor.shape
    # B, A, D = total_tensor.shape
    input_tensor = input_tensor / torch.sqrt(torch.sum(input_tensor ** 2, dim=-1, keepdim=True)).expand(B, P, D)
    # total_tensor = total_tensor / torch.sqrt(torch.sum(total_tensor ** 2, dim=-1, keepdim=True)).expand(B, A, D)
    similarity_matrix = torch.einsum("bpd,bsd->bps", input_tensor, input_tensor)

    # B, X, D = target_tensor.shape
    # B, Z, D = predz.shape
    # target_tensor = target_tensor / torch.sqrt(torch.sum(target_tensor ** 2, dim =-1, keepdim = True)).expand(B,X,D)
    # predz = predz / torch.sqrt(torch.sum(predz ** 2, dim =-1, keepdim = True)).expand(B,Z,D)
    # similarity_matrix_positive = torch.einsum("bxd,bzd->bxz",target_tensor,predz)

    # B, V, P, D = x_enc.shape
        # x_enc_norm = x_enc / torch.sqrt(torch.sum(x_enc ** 2, dim=-1, keepdim=True)).expand(B, V, P, D)
        # B, V, S, D = y.shape
        # y_norm = y / torch.sqrt(torch.sum(y ** 2, dim=-1, keepdim=True)).expand(B, V, S, D)
        # correlation = self.dropout(torch.abs(torch.einsum("bvsd,bvpd->bvsp", y_norm, x_enc_norm)))
        # max_correlation, _ = torch.max(correlation + 1e-6, dim=-1)
        # loss_corr = -torch.log(max_correlation)
        # loss_corr = torch.mean(loss_corr)
    # print('similarity_matrix',similarity_matrix.shape)
    # exit(-1)
    similarity_matrix = torch.exp(similarity_matrix)
    # similarity_matrix_positive = torch.exp(similarity_matrix_positive)

    mask_sim = similarity_matrix
    loss_total = 0
    for j in range(batch_size):
        if patches_numbers[j] == 0:
            continue
        positive_patches = patches_numbers[j]
        choose_patch = np.random.choice([ii for ii in range(total_patch_number-positive_patches)], positive_patches)
        # print('choose_patch',choose_patch + positive_patches)
        # print('mask_sim',mask_sim[j,0,:])
        pos_matrix  = mask_sim[j,0:positive_patches,0:positive_patches]
        pos_matrix = torch.triu(pos_matrix,diagonal = 1)

        pos_matrix = pos_matrix.contiguous().view(-1)
        # pos_matrix_xz = mask_sim[j,0:positive_patches,P:A].contiguous().view(-1)
        # pos_matrix_xz = similarity_matrix_positive[j,:,:].contiguous().view(-1)
        if positive_patches < total_patch_number - positive_patches:
            neg_matrix = mask_sim[j,0:positive_patches,positive_patches + choose_patch].contiguous().view(-1)
        else:
            neg_matrix = mask_sim[j,0:positive_patches,positive_patches:].contiguous().view(-1)
        # print(neg_matrix.shape,neg_matrix[0,:])
        # exit(-1)
        pos = torch.sum(pos_matrix,dim=0)
        # pos_xz = torch.sum(pos_matrix_xz,dim=0)
        neg = torch.sum(neg_matrix,dim=0)

        nominator = pos 
        denominator = pos + neg 
        loss_partial = -torch.log(nominator/denominator)
        loss_total += loss_partial
        
    loss_final = loss_total / batch_size

    return loss_final

class OTETrackActor(BaseActor):
    """ Actor for training OTETrack models """

    def __init__(self, net, objective, loss_weight, settings, bins, search_size, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg
        self.bins = bins
        self.range = self.cfg.MODEL.RANGE
        self.search_size = search_size
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.focal = None
        self.loss_weight['KL'] = 100
        self.loss_weight['focal'] = 2
        self.pre_num = self.cfg.DATA.SEARCH.NUMBER -1

    def __call__(self, data,flag = 0,seq_feat = None, x_feat = None):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        if flag == 0:
            out_dict = self.forward_pass(data)

            # compute losses
            loss, status = self.compute_losses(out_dict, data,flag = 0)
            
            return loss, status

        else:
            out_dict = self.forward_pass(data,flag=1,seq_feat = seq_feat, x_feat = x_feat)
            # compute losses
            # print('1',out_dict['feat'])
            loss, status = self.compute_losses(out_dict, data,flag = 1)

            return loss, status

    def forward_pass(self, data,flag = 0,seq_feat = None, x_feat = None):
        # currently only support 1 template and 1 search region
        # assert len(data['template_images']) == 1
        # assert len(data['search_images']) == 1
        
        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                            *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            template_list.append(template_img_i)

        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)
        # print('search_img',search_img.shape)

        if len(template_list) == 1:
            template_list = template_list[0]

        # gt_bbox = data['search_anno'][-1]
        pre_seq = data['pre_seq']
        # print('pre_seq',pre_seq.shape)

        true_box = data['search_anno'][-1]
        # print('gt_bbox',gt_bbox.shape)
        begin = self.bins * self.range
        end = self.bins * self.range + 1

        magic_num = (self.range - 1) * 0.5
        # gt_bbox[:, 2] = gt_bbox[:, 0] + gt_bbox[:, 2]
        # gt_bbox[:, 3] = gt_bbox[:, 1] + gt_bbox[:, 3]
        # gt_bbox[:, 0] = gt_bbox[:, 0] + gt_bbox[:, 2]/2
        # gt_bbox[:, 1] = gt_bbox[:, 1] + gt_bbox[:, 3]/2
        true_box[:, 2] = true_box[:, 0] + true_box[:, 2]
        true_box[:, 3] = true_box[:, 1] + true_box[:, 3]
        #归一化的坐标
        # gt_bbox = gt_bbox.clamp(min=(-1*magic_num), max=(1+magic_num))
        pre_seq =  pre_seq.clamp(min=(-1*magic_num), max=(1+magic_num))
        # print('pre_seq_after',pre_seq[0,:])
        true_box = true_box.clamp(min=(-1*magic_num), max=(1+magic_num))
        # data['real_bbox'] = true_box
        seq_ori = (pre_seq + magic_num) * (self.bins - 1)
        seq_ori = seq_ori.int().to(search_img)

        seq_true = (true_box + magic_num) * (self.bins - 1)
        seq_true = seq_true.int().to(search_img)
        B = seq_ori.shape[0]
        # seq_input = torch.cat([torch.ones((B, 1)).to(search_img) * begin, seq_ori], dim=1)
        # seq_output = torch.cat([seq_ori, torch.ones((B, 1)).to(search_img) * end], dim=1)
        seq_input = torch.tensor(seq_ori)
        # b * (5x*3+5y*3) 
        seq_output = torch.tensor(seq_true)
        # print(type())
        # print('template_list',template_list.shape)
        # print('search_img',search_img.shape)
        data['seq_input'] = seq_input
        # print('seq_input',seq_input.shape)
        data['seq_output'] = seq_output
        # print('seq_input',seq_input)
        # print('seq_output',seq_output)
        # print('seq.shape',seq_input.shape)
        # seq_input = None
        if flag == 0:
            out_dict = self.net(template=template_list,
                                search=search_img,
                                seq_input=seq_input,
                                flag = 0)
        else:
            # print('flag',flag)
            out_dict = self.net(seq_input=seq_input,
            seq = seq_feat,seq_emd = x_feat,flag = 1)
            print('out_dict',out_dict['feat'])

        return out_dict
                
    def compute_losses(self, pred_dict, gt_dict, return_status=True,flag = 0,seq_feat = None, x_feat = None):
        # print('attention_box',gt_dict['attention_box'])
        # exit(-1)
        bins = self.bins
        magic_num = (self.range - 1) * 0.5
        seq_output = gt_dict['seq_output']
        pred_feat = pred_dict["feat"]
        
        # pred_feat_seq = pred_dict["img_feat"]

        if self.focal == None:
            weight = torch.ones(bins*self.range+2) * 1
            weight[bins*self.range+1] = 0.1
            weight[bins*self.range] = 0.1
            weight.to(pred_feat)
            self.klloss = torch.nn.KLDivLoss(reduction='none').to(pred_feat)

            self.focal = torch.nn.CrossEntropyLoss(weight=weight, size_average=True).to(pred_feat)
        # compute varfifocal loss
        pred = pred_feat.permute(1, 0, 2).reshape(-1, bins*2+2)
        # pred_seq = pred_feat_seq.permute(1, 0, 2).reshape(-1, bins*2+2)
        target = seq_output.reshape(-1).to(torch.int64)

        varifocal_loss = self.focal(pred, target)
        # varifocal_loss_seq = self.focal(pred_seq, target)
        # compute giou and L1 loss
        beta = 1
        pred = pred_feat[0:4, :, 0:bins*self.range] * beta
        # pred_seq = pred_feat_seq[0:4, :, 0:bins*self.range] * beta
        # print('pred_feat',pred_feat.shape)
        target = seq_output[:, 0:4].to(pred_feat)
        
        out = pred.softmax(-1).to(pred)
        # out_seq = pred_seq.softmax(-1).to(pred)
        # print('pred',pred.shape)
        # mul = torch.range((-1*magic_num+1/(self.bins*self.range)), (1+magic_num-1/(self.bins*self.range)), 2/(self.bins*self.range)).to(pred)
        # print(mul)
        mul = torch.range((-1*magic_num), (1+magic_num), 2/(self.bins*self.range))[:-1].to(pred)
        # print('mul',len(mul))
        # exit(-1)
        ans = out * mul
        ans = ans.sum(dim=-1)
        ans = ans.permute(1, 0).to(pred)

        # ans_seq = out_seq * mul
        # ans_seq = ans_seq.sum(dim=-1)
        # ans_seq = ans_seq.permute(1, 0).to(pred)

        target = target / (bins - 1) - magic_num

        extra_seq = ans
        extra_seq = extra_seq.to(pred)

        # extra_seq_1 = ans_seq
        # extra_seq_1 = extra_seq_1.to(pred)
        # print('extra_seq',extra_seq.shape)
        # print('target',target.shape)
        # exit(-1)
        sious, iou = SIoU_loss(extra_seq, target, 4)
        # sious_img, iou_img = SIoU_loss(extra_seq_1, target, 4)
        # sious, iou = SIoU_loss_center(extra_seq, target, 4)
        sious = sious.mean()
        siou_loss = sious

        # sious_img = sious_img.mean()
        # siou_loss_img = sious_img

        l1_loss = self.objective['l1'](extra_seq, target)
        # print('l1_loss',l1_loss)
        # print('attention_IOU_loss11',attention_IOU_loss.item())
        contra_loss_weight = 1

        loss = self.loss_weight['giou'] * siou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * varifocal_loss 
        # + self.loss_weight['focal'] * varifocal_loss_seq + self.loss_weight['giou'] * siou_loss_img
        # + contra_loss_weight*Contra_loss
        # print('loss',loss.item())
        # exit(-1)
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            # mean_iou_img = iou_img.detach().mean()
            if flag == 0:  
                status = {"Loss/total": loss.item(),
                      "Loss/giou": siou_loss.item(),
                    #   "Loss/giou_img": siou_loss_img.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": varifocal_loss.item(),
                    #   "Loss/location_img": varifocal_loss_seq.item(),
                      "IoU": mean_iou.item(),}
                    #   "IoU_img": mean_iou_img.item(),}
                    #   "Contra_loss":Contra_loss.item()}
            else:
                status = {"Loss/total": loss.item(),
                    #   "Loss/giou": siou_loss.item(),
                      "Loss/giou_seq": siou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location_seq": varifocal_loss.item(),
                    #   "Loss/location_img": varifocal_loss_seq.item(),
                      "IoU_seq": mean_iou.item(),}
                   
            return loss, status
        
        else:
            return loss
