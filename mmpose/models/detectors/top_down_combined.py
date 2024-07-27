# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn

import mmcv
import numpy as np
from mmcv.image import imwrite
from mmcv.utils.misc import deprecated_api_warning
from mmcv.visualization.image import imshow

from mmpose.core import imshow_bboxes, imshow_keypoints
from .. import builder
from ..builder import POSENETS
from .base import BasePose

try:
    from mmcv.runner import auto_fp16
except ImportError:
    warnings.warn('auto_fp16 from mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from mmpose.core import auto_fp16


@POSENETS.register_module()
class TopDownCombined(BasePose):
    """Top-down pose detectors.

    Args:
        backbone (dict): Backbone modules to extract feature.
        keypoint_head (dict): Keypoint head to process feature.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path to the pretrained models.
        loss_pose (None): Deprecated arguments. Please use
            `loss_keypoint` for heads instead.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 keypoint_head=None,
                 associate_keypoint_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 loss_pose=None):
        super().__init__()
        self.fp16_enabled = False

        self.backbone = builder.build_backbone(backbone)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if neck is not None:
            self.neck = builder.build_neck(neck)

        if keypoint_head is not None:
            keypoint_head['train_cfg'] = train_cfg
            keypoint_head['test_cfg'] = test_cfg

            if 'loss_keypoint' not in keypoint_head and loss_pose is not None:
                warnings.warn(
                    '`loss_pose` for TopDown is deprecated, '
                    'use `loss_keypoint` for heads instead. See '
                    'https://github.com/open-mmlab/mmpose/pull/382'
                    ' for more information.', DeprecationWarning)
                keypoint_head['loss_keypoint'] = loss_pose

            self.keypoint_head = builder.build_head(keypoint_head)
        
        # Add associate heads
        associate_keypoint_heads = []
        keypoint_heads_cnt = 1

        if associate_keypoint_head is not None:
            if not isinstance(associate_keypoint_head, list):
                associate_keypoint_head = [associate_keypoint_head]
            for single_keypoint_head in associate_keypoint_head:
                single_keypoint_head['train_cfg'] = train_cfg
                single_keypoint_head['test_cfg'] = test_cfg
                associate_keypoint_heads.append(builder.build_head(single_keypoint_head))
                keypoint_heads_cnt += 1

        self.associate_keypoint_heads = nn.ModuleList(associate_keypoint_heads)

        self.keypoint_heads_cnt = keypoint_heads_cnt

        self.init_weights(pretrained=pretrained)

    @property
    def with_neck(self):
        """Check if has neck."""
        return hasattr(self, 'neck')

    @property
    def with_keypoint(self):
        """Check if has keypoint_head."""
        return hasattr(self, 'keypoint_head')

    def init_weights(self, pretrained=None):
        """Weight initialization for model."""
        self.backbone.init_weights(pretrained)
        if self.with_neck:
            self.neck.init_weights()
        if self.with_keypoint:
            self.keypoint_head.init_weights()
        for item in self.associate_keypoint_heads:
            item.init_weights()

    @auto_fp16(apply_to=('img', ))
    def forward(self,
                img,
                target=None,
                target_weight=None,
                img_metas=None,
                return_loss=True,
                return_heatmap=False,
                **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True. Note this setting will change the expected inputs.
        When `return_loss=True`, img and img_meta are single-nested (i.e.
        Tensor and List[dict]), and when `resturn_loss=False`, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.

        Note:
            - batch_size: N
            - num_keypoints: K
            - num_img_channel: C (Default: 3)
            - img height: imgH
            - img width: imgW
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            img (torch.Tensor[NxCximgHximgW]): Input images.
            target (torch.Tensor[NxKxHxW]): Target heatmaps.
            target_weight (torch.Tensor[NxKx1]): Weights across
                different joint types.
            img_metas (list(dict)): Information about data augmentation
                By default this includes:

                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
            return_loss (bool): Option to `return loss`. `return loss=True`
                for training, `return loss=False` for validation & test.
            return_heatmap (bool) : Option to return heatmap.

        Returns:
            dict|tuple: if `return loss` is true, then return losses. \
                Otherwise, return predicted poses, boxes, image paths \
                and heatmaps.
        """
        if return_loss:
            return self.forward_train(img, target, target_weight, img_metas,
                                      **kwargs)
        return self.forward_test(
            img, img_metas, return_heatmap=return_heatmap, **kwargs)


    def reverse_row(row):
        return row[::-1]
    # TODO flow multi-head losses. But training procedure seems different from MoE
    def forward_train(self, img, target, target_weight, img_metas, **kwargs):
        """Defines the computation performed at every call when training."""

        output = self.backbone(img)
        if self.with_neck:
            output = self.neck(output)
        if self.with_keypoint:
            output = self.keypoint_head(output)
        img_sources = torch.from_numpy(np.array([ele['dataset_idx'] for ele in img_metas])).to(img.device)
        
        # mpii_places = []
        
        # Image target coming as 33 - defined by max_num_joints in config
        # to propogate losses in their respective correct places
        # ones has to structure keypoints so that they flow loss to their correct places
        # as a default target is like padded tensor with either first 16 or 17 filled
        # their decision to hav max num joints makes sense as they can't have
        # tensor with Nx(17|16) elements so they decide to go with max_num_joints
        # for my decision, I keep 17 points on their place if COCO
        # otherwise put this 16 points in the last
        coco_targets = (img_sources == 0).view(-1,1,1,1)
        mpii_targets = (img_sources == 1).view(-1,1,1,1)

        coco_select = target * coco_targets
        mpii_select = target * mpii_targets
        target_select = coco_select + mpii_select.flip(dims=[1])
        target_weight[:, :17,0] = 1
        target_weight[:, 17:,0] = 0
        coco_target_weight_select = target_weight * (img_sources == 0).view(-1,1,1)
        mpii_target_weight_select = target_weight * (img_sources == 1).view(-1,1,1)
        mpii_target_weight_select_flipped = mpii_target_weight_select.flip(dims=[1])
        mpii_target_weight_select_flipped[:, 17,0] = 0
        target_weight_select = coco_target_weight_select + mpii_target_weight_select_flipped
        # print(target_weight_select)
        # 33 
        # first 17 coco and last 16 is mpii but reversed
        # had to be done cuz of padding to 33 joints for each training batch
        


        # # if return loss
        # batch_size = target.size(0)
        # num_keypoint = 17
        # num_keypoint_mpii = 16
        # num_keypoint_coco = 16
        # # N x 133 x W x H -> 
        # new_target = torch.zeros((batch_size, 
        #                           num_keypoint_mpii + num_keypoint_coco,
        #                           target.size(2),
        #                           target.size(3)
        #                           )).to(img.device)
        # new_target[:, ] = target[:,]
        # Output -> N x 33 x 2
        # if ()
        # output -> [N x 33 x 2]
        # depending on the img_metas, only calculate loss for that set of indexes.

        losses = dict()
        if self.with_keypoint:
            keypoint_losses = self.keypoint_head.get_loss(
                output, target_select, target_weight_select)
            losses.update(keypoint_losses)
            keypoint_accuracy = self.keypoint_head.get_accuracy(
                output, target_select, target_weight_select)
            losses.update(keypoint_accuracy)

        return losses

    def forward_test(self, img, img_metas, return_heatmap=False, **kwargs):
        """Defines the computation performed at every call when testing."""
        assert img.size(0) == len(img_metas)
        batch_size, _, img_height, img_width = img.shape
        if batch_size > 1:
            assert 'bbox_id' in img_metas[0]

        result = {}

        features = self.backbone(img)
        if self.with_neck:
            features = self.neck(features)
        if self.with_keypoint:
            output_heatmap = self.keypoint_head.inference_model(
                features, flip_pairs=None)

        # if self.test_cfg.get('flip_test', True):
        #     img_flipped = img.flip(3)
        #     features_flipped = self.backbone(img_flipped)
        #     if self.with_neck:
        #         features_flipped = self.neck(features_flipped)
        #     if self.with_keypoint:
        #         output_flipped_heatmap = self.keypoint_head.inference_model(
        #             features_flipped, img_metas[0]['flip_pairs'])
        #         output_heatmap = (output_heatmap +
        #                           output_flipped_heatmap) * 0.5

        if self.with_keypoint:
            keypoint_result = self.keypoint_head.decode(
                img_metas, output_heatmap, img_size=[img_width, img_height])
            result.update(keypoint_result)

            if not return_heatmap:
                output_heatmap = None

            result['output_heatmap'] = output_heatmap

        return result

    def forward_dummy(self, img):
        """Used for computing network FLOPs.

        See ``tools/get_flops.py``.

        Args:
            img (torch.Tensor): Input image.

        Returns:
            Tensor: Output heatmaps.
        """
        output = self.backbone(img)
        if self.with_neck:
            output = self.neck(output)
        if self.with_keypoint:
            output = self.keypoint_head(output)
        return output

    @deprecated_api_warning({'pose_limb_color': 'pose_link_color'},
                            cls_name='TopDown')
    def show_result(self,
                    img,
                    result,
                    skeleton=None,
                    kpt_score_thr=0.3,
                    bbox_color='green',
                    pose_kpt_color=None,
                    pose_link_color=None,
                    text_color='white',
                    radius=4,
                    thickness=1,
                    font_scale=0.5,
                    bbox_thickness=1,
                    win_name='',
                    show=False,
                    show_keypoint_weight=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (list[dict]): The results to draw over `img`
                (bbox_result, pose_result).
            skeleton (list[list]): The connection of keypoints.
                skeleton is 0-based indexing.
            kpt_score_thr (float, optional): Minimum score of keypoints
                to be shown. Default: 0.3.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            pose_kpt_color (np.array[Nx3]`): Color of N keypoints.
                If None, do not draw keypoints.
            pose_link_color (np.array[Mx3]): Color of M links.
                If None, do not draw links.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            radius (int): Radius of circles.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            show (bool): Whether to show the image. Default: False.
            show_keypoint_weight (bool): Whether to change the transparency
                using the predicted confidence scores of keypoints.
            wait_time (int): Value of waitKey param.
                Default: 0.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            Tensor: Visualized img, only if not `show` or `out_file`.
        """
        img = mmcv.imread(img)
        img = img.copy()

        bbox_result = []
        bbox_labels = []
        pose_result = []
        for res in result:
            if 'bbox' in res:
                bbox_result.append(res['bbox'])
                bbox_labels.append(res.get('label', None))
            pose_result.append(res['keypoints'])

        if bbox_result:
            bboxes = np.vstack(bbox_result)
            # draw bounding boxes
            imshow_bboxes(
                img,
                bboxes,
                labels=bbox_labels,
                colors=bbox_color,
                text_color=text_color,
                thickness=bbox_thickness,
                font_scale=font_scale,
                show=False)

        if pose_result:
            imshow_keypoints(img, pose_result, skeleton, kpt_score_thr,
                             pose_kpt_color, pose_link_color, radius,
                             thickness)

        if show:
            imshow(img, win_name, wait_time)

        if out_file is not None:
            imwrite(img, out_file)

        return img