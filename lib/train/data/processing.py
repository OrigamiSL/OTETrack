import torch
import torchvision.transforms as transforms
from lib.utils import TensorDict
import lib.train.data.processing_utils as prutils
import torch.nn.functional as F
import lib.train.data.transforms as tfm
import random

def stack_tensors(x):
    if isinstance(x, (list, tuple)) and isinstance(x[0], torch.Tensor):
        return torch.stack(x)
    return x


class BaseProcessing:
    """ Base class for Processing. Processing class is used to process the data returned by a dataset, before passing it
     through the network. For example, it can be used to crop a search region around the object, apply various data
     augmentations, etc."""
    def __init__(self, transform=transforms.ToTensor(), template_transform=None, search_transform=None, joint_transform=None):
        """
        args:
            transform       - The set of transformations to be applied on the images. Used only if template_transform or
                                search_transform is None.
            template_transform - The set of transformations to be applied on the template images. If None, the 'transform'
                                argument is used instead.
            search_transform  - The set of transformations to be applied on the search images. If None, the 'transform'
                                argument is used instead.
            joint_transform - The set of transformations to be applied 'jointly' on the template and search images.  For
                                example, it can be used to convert both template and search images to grayscale.
        """
        self.transform = {'template': transform if template_transform is None else template_transform,
                          'search':  transform if search_transform is None else search_transform,
                          'joint': joint_transform}

    def __call__(self, data: TensorDict):
        raise NotImplementedError


class STARKProcessing(BaseProcessing):
    """ The processing class used for training LittleBoy. The images are processed in the following way.
    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument output_sz.

    """

    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor,if_seq,
                 mode='pair', settings=None, *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.mode = mode
        self.settings = settings
        self.if_seq = if_seq
    #     transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),
    #                                 tfm.RandomHorizontalFlip(probability=0))

    # # transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
    # #                                 tfm.RandomHorizontalFlip_Norm(probability=0.5),
    # #                                 tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))
    #     transform_train = tfm.Transform(
    #                                     tfm.RandomHorizontalFlip_Norm(probability=0),                                  
    #                                     )
    def _get_jittered_box(self, box, mode,box_true=None):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'template' or 'search' indicating template or search data

        returns:
            torch.Tensor - jittered box
        """
        # print('box',box)
        # box = torch.tensor(box)
        # if mode == 'template':
        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        # 68%  0.779 - 1.28
        # 95%  0.607- 1.65
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)
        # else:
        #     jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        #     max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        #     jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)#(-0.5,1.5 max_offset)

        #     return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)


    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'template_images', search_images', 'template_anno', 'search_anno'
        returns:
            TensorDict - output data block with following fields:
                'template_images', 'search_images', 'template_anno', 'search_anno', 'test_proposals', 'proposal_iou'
        """
        # Apply joint transforms

        if self.if_seq:
            if self.transform['joint'] is not None:
                data['template_images'], data['template_anno'], data['template_masks'] = self.transform['joint'](
                    image=data['template_images'], bbox=data['template_anno'], mask=data['template_masks'])
                
                data['search_images'], data['search_anno'], data['search_masks'] = self.transform['joint'](
                    image=data['search_images'], bbox=data['search_anno'], mask=data['search_masks'], new_roll=False)
                
                boxes_joint = []

                # for frame in data['previous_anno']:
                #     # _, box_joint, _ = self.transform['joint'](
                #     # image=data['search_images'], bbox=frame, mask=data['search_masks'], new_roll=False)
                #     # boxes_joint.append(box_joint)
                #     box_joint= self.transform['joint'].transforms[1](image=data['search_images'],
                #     bbox=frame, new_roll=False)['bbox']
                #     boxes_joint.append(box_joint)
                #     # print('box_joint',box_joint)
                # # exit(-1)
                # data['previous_anno'] = boxes_joint

            for s in ['template', 'search']:
            
                assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                    "In pair mode, num train/test frames must be 1"
                # print(s)
                # print(s + '_anno',data[s + '_anno'])
                # print(s + '_anno',data[s + '_anno'])
                # Add a uniform noise to the center pos
                if s == 'template':
                    jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

                    # 2021.1.9 Check whether data is valid. Avoid too small bounding boxes
                    w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]

                    crop_sz = torch.ceil(torch.sqrt(w * h) * self.search_area_factor[s])
                    if (crop_sz < 1).any():
                        
                        data['valid'] = False
                        # print("Too small box is found. Replace it with new data.")
                        return data
                    # Crop image region centered at jittered_anno box and get the attention mask
                    crops, boxes, att_mask, mask_crops,_ = prutils.jittered_center_crop(data[s + '_images'], jittered_anno,
                                                                                    data[s + '_anno'], self.search_area_factor[s],
                                                                                    self.output_sz[s], masks=data[s + '_masks'])
                    # Apply transforms
                    data[s + '_images'], data[s + '_anno'], data[s + '_att'], data[s + '_masks'] = self.transform[s](
                        image=crops, bbox=boxes, att=att_mask, mask=mask_crops, joint=False)
                else:
                    # print('s_test',data[s + '_anno'])

                    # judge moving 
                    # x1,y1,w,h = data[s + '_anno'][0]
                    # x1_p,y1_p,w_p,h_p = data['previous_anno'][-1]
                    # xc,yc = x1 + w/2, y1 + h/2
                    # xc_p,yc_p = x1_p + w_p/2, y1_p + h_p/2
                    # if abs(xc - xc_p) > w_p or abs(yc - yc_p) > h_p:
                    #     move_fast = True
                    # else:
                    #     move_fast = False

                    # for i in range(len(data['previous_anno'])):
                    #     if i == len(data['previous_anno']) -1 :
                    #     else:
                    jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]
                    # pre_in_last[0 :2] = pre_seq_last[i, 0 :2] - self.center_pos[i]
                    # pre_in_last[2 :4 ] = pre_seq_last[i, 2 :4 ] - self.center_pos[i]
                    # pre_in_last[0 :4 ] = pre_in_last[0 :4 ] * (
                    #             self.cfg.DATA.SEARCH.SIZE / s_x[i]) + self.cfg.DATA.SEARCH.SIZE / 2
                    # pre_in_last[0 :4 ] = pre_in_last[0 :4 ] / self.cfg.DATA.SEARCH.SIZE
                    # jittered_anno = [self._get_jittered_box(data['search_anno'], 'search',data['true_anno'])]
                    # 2021.1.9 Check whether data is valid. Avoid too small bounding boxes
                    w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]

                    crop_sz = torch.ceil(torch.sqrt(w * h) * self.search_area_factor[s])

                    if (crop_sz < 1).any():
                        
                        data['valid'] = False
                        # print("Too small box is found. Replace it with new data.")
                        return data
                    # Crop image region centered at jittered_anno box and get the attention mask
                    # crops, boxes, att_mask, mask_crops = prutils.jittered_center_crop(data[s + '_images'], jittered_anno,
                    #                                                                 data[s + '_anno'], self.search_area_factor[s],
                    #                                                                 self.output_sz[s], masks=data[s + '_masks'])
                    
                    crops, boxes, att_mask, mask_crops,resize_factors = prutils.jittered_center_crop(data[s + '_images'], jittered_anno,
                                                                                    data[s + '_anno'], self.search_area_factor[s],
                                                                                  self.output_sz[s], masks=data[s + '_masks'])
                    boxes_pre = []

                    # crop_sz_input = torch.Tensor([self.output_sz[s], self.output_sz[s]])
                    # for frame in data['previous_anno']:
                    #     frame = [frame]
                    #     # _, box_pre, _, _ = prutils.jittered_center_crop(data[s + '_images'], jittered_anno,
                    #     #                                                             frame, self.search_area_factor[s],
                    #     #  
                    #     #                                                            self.output_sz[s], masks=data[s + '_masks'])
                         
                    #     box_pre = [prutils.transform_image_to_crop(a_gt, a_ex, None, rf, crop_sz_input, normalize=True,use_predict=False)
                    #             for a_gt, a_ex, rf in zip(frame, jittered_anno, resize_factors)]
                    #     boxes_pre.append(box_pre)
                    
                    # Apply transforms
                    # print('before',boxes,boxes_true)
                    # print('start')
                    data[s + '_images'], data[s + '_anno'], data[s + '_att'], data[s + '_masks'] = self.transform[s](
                        image=crops, bbox=boxes, att=att_mask, mask=mask_crops, joint=False)
                    
                    # data_result = []
                    # for i in range(len(data['previous_anno'])):
                    #     # _, result, _, _ = self.transform[s](
                    #     #     image=crops, bbox=boxes_pre[i], att=att_mask, mask=mask_crops, joint=False,new_roll = False)
                    #     # data_result.append(result[0])
                    #     result = self.transform[s].transforms[1](
                    #         image=crops, bbox=boxes_pre[i], joint=False,new_roll = False)['bbox']
                    #     data_result.append(result[0])
                        
                    
                    # data['previous_anno'] = data_result


                    # data['true_anno'] = boxes_true

                    # print('af',data[s + '_anno'],data['true_anno'])

                    # data[s + '_images'], data[s + '_anno'], data[s + '_att'], data[s + '_masks'] = self.transform[s](
                    #     image=crops, bbox=boxes, att=att_mask, mask=mask_crops, joint=False)
                # 2021.1.9 Check whether elements in data[s + '_att'] is all 1
                # Note that type of data[s + '_att'] is tuple, type of ele is torch.tensor
                for ele in data[s + '_att']:
                    
                    if (ele == 1).all():
                        data['valid'] = False
                        # print("Values of original attention mask are all one. Replace it with new data.")
                        return data
                # 2021.1.10 more strict conditions: require the donwsampled masks not to be all 1
                # for ele in data[s + '_att']:
                    
                #     feat_size = self.output_sz[s] // 16  # 16 is the backbone stride
                #     # (1,1,128,128) (1,1,256,256) --> (1,1,8,8) (1,1,16,16)
                #     mask_down = F.interpolate(ele[None, None].float(), size=feat_size).to(torch.bool)[0]
                #     if (mask_down == 1).all():
                #         data['valid'] = False
                #         # print("Values of down-sampled attention mask are all one. "
                #         #       "Replace it with new data.")
                #         return data
            
            data['valid'] = True
            # if we use copy-and-paste augmentation
            if data["template_masks"] is None or data["search_masks"] is None:
                data["template_masks"] = torch.zeros((1, self.output_sz["template"], self.output_sz["template"]))
                data["search_masks"] = torch.zeros((1, self.output_sz["search"], self.output_sz["search"]))
            # Prepare output
            if self.mode == 'sequence':
                data = data.apply(stack_tensors)
            else:
                data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

            return data
        
        else:
            if self.transform['joint'] is not None:
                data['template_images'], data['template_anno'], data['template_masks'] = self.transform['joint'](
                    image=data['template_images'], bbox=data['template_anno'], mask=data['template_masks'])
                data['search_images'], data['search_anno'], data['search_masks'] = self.transform['joint'](
                    image=data['search_images'], bbox=data['search_anno'], mask=data['search_masks'], new_roll=False)

            for s in ['template', 'search']:
                assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                    "In pair mode, num train/test frames must be 1"
                
                # if s == 'search':
                #     print('b',data['search_anno'])

                # Add a uniform noise to the center pos
                jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

                # 2021.1.9 Check whether data is valid. Avoid too small bounding boxes
                w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]

                crop_sz = torch.ceil(torch.sqrt(w * h) * self.search_area_factor[s])
                if (crop_sz < 1).any():
                    data['valid'] = False
                    # print("Too small box is found. Replace it with new data.")
                    return data

                # Crop image region centered at jittered_anno box and get the attention mask
                crops, boxes, att_mask, mask_crops = prutils.jittered_center_crop(data[s + '_images'], jittered_anno,
                                                                                data[s + '_anno'], self.search_area_factor[s],
                                                                                self.output_sz[s], masks=data[s + '_masks'])
                
                # Apply transforms
                data[s + '_images'], data[s + '_anno'], data[s + '_att'], data[s + '_masks'] = self.transform[s](
                    image=crops, bbox=boxes, att=att_mask, mask=mask_crops, joint=False)

                # if s == 'search':
                #     print('a',data['search_anno'])

                # 2021.1.9 Check whether elements in data[s + '_att'] is all 1
                # Note that type of data[s + '_att'] is tuple, type of ele is torch.tensor
                for ele in data[s + '_att']:
                    if (ele == 1).all():
                        data['valid'] = False
                        # print("Values of original attention mask are all one. Replace it with new data.")
                        return data
                # # 2021.1.10 more strict conditions: require the donwsampled masks not to be all 1
                for ele in data[s + '_att']:
                    feat_size = self.output_sz[s] // 16  # 16 is the backbone stride
                    # (1,1,128,128) (1,1,256,256) --> (1,1,8,8) (1,1,16,16)
                    mask_down = F.interpolate(ele[None, None].float(), size=feat_size).to(torch.bool)[0]
                    if (mask_down == 1).all():
                        data['valid'] = False
                        # print("Values of down-sampled attention mask are all one. "
                        #       "Replace it with new data.")
                        return data

            data['valid'] = True
            # if we use copy-and-paste augmentation
            if data["template_masks"] is None or data["search_masks"] is None:
                data["template_masks"] = torch.zeros((1, self.output_sz["template"], self.output_sz["template"]))
                data["search_masks"] = torch.zeros((1, self.output_sz["search"], self.output_sz["search"]))
            # Prepare output
            if self.mode == 'sequence':
                data = data.apply(stack_tensors)
            else:
                data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

            return data
        

