import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta
import math


class _fasterRCNN(nn.Module):
    """ faster RCNN """

    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0 / 16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0 / 16.0)

        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.RCNN_roi_crop = _RoICrop()

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
        base_feat = self.RCNN_base(im_data)

        avg_feat = self.spatial_pool(base_feat, [base_feat.size()[2], base_feat.size()[3]], 14)

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
        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data
            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == 'crop':
            grid_xy = _affine_grid_gen(rois.view(-1, 5), base_feat.size()[2:], self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:, :, :, 1], grid_xy.data[:, :, :, 0]], 3).contiguous()
            pooled_feat = self.RCNN_roi_crop(base_feat, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
        elif cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1, 5))

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1,
                                            rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

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

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_altitude_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_angle_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_weather_score, 0, 0.01, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
