import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, gradcheck
from torch.autograd.gradcheck import gradgradcheck
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
import torchvision.utils as vutils
from model.utils.config import cfg
from model.rpn.rpn_fpn import _RPN_FPN
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.rpn.proposal_target_layer import _ProposalTargetLayer
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta
import time
import pdb
import math

class _FPN(nn.Module):
    """ FPN """
    def __init__(self, classes, class_agnostic):
        super(_FPN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        self.maxpool2d = nn.MaxPool2d(1, stride=2)
        # define rpn
        self.RCNN_rpn = _RPN_FPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)

        # NOTE: the original paper used pool_size = 7 for cls branch, and 14 for mask branch, to save the
        # computation time, we first use 14 as the pool_size, and then do stride=2 pooling for cls branch.
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.RCNN_roi_crop = _RoICrop()

    def normal_init(self, m, mean, stddev, truncated=False):
        """
        weight initalizer: truncated normal and random normal.
        """
        # x is a parameter
        if truncated:
            m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
        else:
            m.weight.data.normal_(mean, stddev)
            m.bias.data.zero_()

    # custom weights initialization called on netG and netD
    def weights_init(self, m, mean, stddev, truncated=False):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def _init_weights(self):
        self.normal_init(self.RCNN_toplayer, 0, 0.01, cfg.TRAIN.TRUNCATED)
        self.normal_init(self.RCNN_smooth1, 0, 0.01, cfg.TRAIN.TRUNCATED)
        self.normal_init(self.RCNN_smooth2, 0, 0.01, cfg.TRAIN.TRUNCATED)
        self.normal_init(self.RCNN_smooth3, 0, 0.01, cfg.TRAIN.TRUNCATED)
        self.normal_init(self.RCNN_latlayer1, 0, 0.01, cfg.TRAIN.TRUNCATED)
        self.normal_init(self.RCNN_latlayer2, 0, 0.01, cfg.TRAIN.TRUNCATED)
        self.normal_init(self.RCNN_latlayer3, 0, 0.01, cfg.TRAIN.TRUNCATED)

        self.normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        self.normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        self.normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        self.normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        self.normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)
        self.normal_init(self.RCNN_weather_score, 0, 0.001, cfg.TRAIN.TRUNCATED)
        self.normal_init(self.RCNN_angle_score, 0, 0.001, cfg.TRAIN.TRUNCATED)
        self.normal_init(self.RCNN_altitude_score, 0, 0.001, cfg.TRAIN.TRUNCATED)

        self.weights_init(self.RCNN_top, 0, 0.01, cfg.TRAIN.TRUNCATED)
        self.weights_init(self.RCNN_weather, 0, 0.01, cfg.TRAIN.TRUNCATED)
        self.weights_init(self.RCNN_angle, 0, 0.01, cfg.TRAIN.TRUNCATED)
        self.weights_init(self.RCNN_altitude, 0, 0.01, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def _PyramidRoI_Feat(self, feat_maps, rois, im_info):
        ''' roi pool on pyramid feature maps'''
        # do roi pooling based on predicted rois
        img_area = im_info[0][0] * im_info[0][1]
        h = rois.data[:, 4] - rois.data[:, 2] + 1
        w = rois.data[:, 3] - rois.data[:, 1] + 1
        roi_level = torch.log(torch.sqrt(h * w) / 224.0)
        roi_level = torch.round(roi_level + 4)
        roi_level[roi_level < 2] = 2
        roi_level[roi_level > 5] = 5
        # roi_level.fill_(5)
        if cfg.POOLING_MODE == 'align':
            roi_pool_feats = []
            box_to_levels = []
            for i, l in enumerate(range(2, 6)):
                if (roi_level == l).sum() == 0:
                    continue
                idx_l = (roi_level == l).nonzero()
                if idx_l.shape[0] > 1:
                    idx_l = idx_l.squeeze()
                else:
                    idx_l = idx_l.view(-1)
                box_to_levels.append(idx_l)
                scale = feat_maps[i].size(2) / im_info[0][0]
                feat = self.RCNN_roi_align(feat_maps[i], rois[idx_l], scale)
                roi_pool_feats.append(feat)
            roi_pool_feat = torch.cat(roi_pool_feats, 0)
            box_to_level = torch.cat(box_to_levels, 0)
            idx_sorted, order = torch.sort(box_to_level)
            roi_pool_feat = roi_pool_feat[order]

        elif cfg.POOLING_MODE == 'pool':
            roi_pool_feats = []
            box_to_levels = []
            for i, l in enumerate(range(2, 6)):
                if (roi_level == l).sum() == 0:
                    continue
                idx_l = (roi_level == l).nonzero().squeeze()
                box_to_levels.append(idx_l)
                scale = feat_maps[i].size(2) / im_info[0][0]
                feat = self.RCNN_roi_pool(feat_maps[i], rois[idx_l], scale)
                roi_pool_feats.append(feat)
            roi_pool_feat = torch.cat(roi_pool_feats, 0)
            box_to_level = torch.cat(box_to_levels, 0)
            idx_sorted, order = torch.sort(box_to_level)
            roi_pool_feat = roi_pool_feat[order]
            
        return roi_pool_feat

    def spatial_pool(self, input_conv, input_conv_size, output_pool_size):
        h_wid = int(math.ceil(input_conv_size[0] / output_pool_size))
        w_wid = int(math.ceil(input_conv_size[1] / output_pool_size))
        #h_pad = (h_wid * output_pool_size - input_conv_size[0] + 1) // 2
        #w_pad = (w_wid * output_pool_size - input_conv_size[1] + 1) // 2
        h_pad = int((h_wid * output_pool_size - input_conv_size[0] + 1) / 2)
        w_pad = int((w_wid * output_pool_size - input_conv_size[1] + 1) / 2)
        zero_pad = nn.ZeroPad2d((w_pad, w_pad, h_pad, h_pad))
        maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid))
        #maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
        return maxpool(zero_pad(input_conv))

    def forward(self, im_data, im_info, meta_data, gt_boxes, num_boxes, run_partial=False):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data
        meta_data = meta_data.data

        # feed image data to base model to obtain base feature map
        # Bottom-up
        c1 = self.RCNN_layer0(im_data)
        c2 = self.RCNN_layer1(c1)
        c3 = self.RCNN_layer2(c2)
        c4 = self.RCNN_layer3(c3)
        c5 = self.RCNN_layer4(c4)
        # Top-down
        p5 = self.RCNN_toplayer(c5)
        p4 = self._upsample_add(p5, self.RCNN_latlayer1(c4))
        p4 = self.RCNN_smooth1(p4)
        p3 = self._upsample_add(p4, self.RCNN_latlayer2(c3))
        p3 = self.RCNN_smooth2(p3)
        p2 = self._upsample_add(p3, self.RCNN_latlayer3(c2))
        p2 = self.RCNN_smooth3(p2)

        p6 = self.maxpool2d(p5)

        rpn_feature_maps = [p2, p3, p4, p5, p6]
        mrcnn_feature_maps = [p2, p3, p4, p5]

        base_feat = p2

        avg_feat = self.spatial_pool(base_feat, [base_feat.size()[2], base_feat.size()[3]], 28)

        weather_label = Variable(meta_data[:, 0].view(-1).long())
        altitude_label = Variable(meta_data[:, 1].view(-1).long())
        angle_label = Variable(meta_data[:, 2].view(-1).long())
        softmax = nn.Softmax(dim=1)

        '''
        Altitude
        '''
        altitude_score = self.RCNN_altitude_score(self.RCNN_altitude(avg_feat).mean(-1).mean(-1))
        RCNN_loss_altitude = F.cross_entropy(altitude_score, altitude_label)
        # RCNN_loss_altitude_adv = torch.mean(torch.sum(- altitude_score.new_full(altitude_score.size(), 1 / 3.0) * torch.log(torch.clamp(softmax(altitude_score), min=1e-10, max=1.0)), 1))
        RCNN_loss_altitude_adv = torch.mean(
            torch.sum(softmax(altitude_score) * torch.log(torch.clamp(softmax(altitude_score), min=1e-10, max=1.0)), 1))
        correct_altitude = altitude_score.max(1)[1].type_as(altitude_label).eq(altitude_label)
        correct_altitude = correct_altitude.sum().type(torch.FloatTensor).cuda()
        RCNN_acc_altitude = correct_altitude / altitude_label.size(0)

        '''
        View Angle
        '''
        angle_score = self.RCNN_angle_score(self.RCNN_angle(avg_feat).mean(-1).mean(-1))
        RCNN_loss_angle = F.cross_entropy(angle_score, angle_label)
        RCNN_loss_angle_adv = torch.mean(
            torch.sum(softmax(angle_score) * torch.log(torch.clamp(softmax(angle_score), min=1e-10, max=1.0)), 1))
        correct_angle = angle_score.max(1)[1].type_as(angle_label).eq(angle_label)
        correct_angle = correct_angle.sum().type(torch.FloatTensor).cuda()
        RCNN_acc_angle = correct_angle / angle_label.size(0)

        '''
        Weather
        '''
        weather_score = self.RCNN_weather_score(self.RCNN_weather(avg_feat).mean(-1).mean(-1))
        RCNN_loss_weather = F.cross_entropy(weather_score, weather_label)
        RCNN_loss_weather_adv = torch.mean(
            torch.sum(softmax(weather_score) * torch.log(torch.clamp(softmax(weather_score), min=1e-10, max=1.0)), 1))
        correct_weather = weather_score.max(1)[1].type_as(weather_label).eq(weather_label)
        correct_weather = correct_weather.sum().type(torch.FloatTensor).cuda()
        RCNN_acc_weather = correct_weather / weather_label.size(0)
        if run_partial:
            if self.training:
                RCNN_loss_altitude = torch.unsqueeze(RCNN_loss_altitude, 0)
                RCNN_loss_altitude_adv = torch.unsqueeze(RCNN_loss_altitude_adv, 0)
                RCNN_acc_altitude = torch.unsqueeze(RCNN_acc_altitude, 0)

                RCNN_loss_angle = torch.unsqueeze(RCNN_loss_angle, 0)
                RCNN_loss_angle_adv = torch.unsqueeze(RCNN_loss_angle_adv, 0)
                RCNN_acc_angle = torch.unsqueeze(RCNN_acc_angle, 0)

                RCNN_loss_weather = torch.unsqueeze(RCNN_loss_weather, 0)
                RCNN_loss_weather_adv = torch.unsqueeze(RCNN_loss_weather_adv, 0)
                RCNN_acc_weather = torch.unsqueeze(RCNN_acc_weather, 0)
            return RCNN_loss_altitude, RCNN_loss_altitude_adv, RCNN_acc_altitude, \
                   RCNN_loss_angle, RCNN_loss_angle_adv, RCNN_acc_angle, \
                   RCNN_loss_weather, RCNN_loss_weather_adv, RCNN_acc_weather




        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(rpn_feature_maps, im_info, gt_boxes, num_boxes)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, gt_assign, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois = rois.view(-1, 5)
            rois_label = rois_label.view(-1).long()
            gt_assign = gt_assign.view(-1).long()
            pos_id = rois_label.nonzero().squeeze()

            rois = Variable(rois)
            rois_label = Variable(rois_label)

            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            gt_assign = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0
            rois = rois.view(-1, 5)
            pos_id = torch.arange(0, rois.size(0)).long().type_as(rois).long()
            rois = Variable(rois)

        # pooling features based on rois, output 14x14 map
        roi_pool_feat = self._PyramidRoI_Feat(mrcnn_feature_maps, rois, im_info)

        #print(roi_pool_feat.shape)
        # feed pooled features to top model
        pooled_feat = self._head_to_tail(roi_pool_feat)


        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.long().view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # loss (cross entropy) for object classification
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)
            # loss (l1-norm) for bounding box regression
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        rois = rois.view(batch_size, -1, rois.size(1))
        cls_prob = cls_prob.view(batch_size, -1, cls_prob.size(1))
        bbox_pred = bbox_pred.view(batch_size, -1, bbox_pred.size(1))

        if self.training:
            rois_label = rois_label.view(batch_size, -1)

        if self.training:
            rpn_loss_cls = torch.unsqueeze(rpn_loss_cls, 0)
            rpn_loss_bbox = torch.unsqueeze(rpn_loss_bbox, 0)
            RCNN_loss_cls = torch.unsqueeze(RCNN_loss_cls, 0)
            RCNN_loss_bbox = torch.unsqueeze(RCNN_loss_bbox, 0)

            RCNN_loss_altitude = torch.unsqueeze(RCNN_loss_altitude, 0)
            RCNN_loss_altitude_adv = torch.unsqueeze(RCNN_loss_altitude_adv, 0)
            RCNN_acc_altitude = torch.unsqueeze(RCNN_acc_altitude, 0)

            RCNN_loss_angle = torch.unsqueeze(RCNN_loss_angle, 0)
            RCNN_loss_angle_adv = torch.unsqueeze(RCNN_loss_angle_adv, 0)
            RCNN_acc_angle = torch.unsqueeze(RCNN_acc_angle, 0)

            RCNN_loss_weather = torch.unsqueeze(RCNN_loss_weather, 0)
            RCNN_loss_weather_adv = torch.unsqueeze(RCNN_loss_weather_adv, 0)
            RCNN_acc_weather = torch.unsqueeze(RCNN_acc_weather, 0)

            return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, \
                   RCNN_loss_altitude, RCNN_loss_altitude_adv, RCNN_acc_altitude, \
                   RCNN_loss_angle, RCNN_loss_angle_adv, RCNN_acc_angle, \
                   RCNN_loss_weather, RCNN_loss_weather_adv, RCNN_acc_weather, \
                   rois_label

        return rois, cls_prob, bbox_pred, RCNN_acc_altitude, RCNN_acc_angle, RCNN_acc_weather