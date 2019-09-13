# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
    adjust_learning_rate, save_checkpoint, clip_gradient

from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
import itertools
import math

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='uav', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res101',
                        default='res101', type=str)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=20, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='number of iterations to display',
                        default=10, type=int)
    parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                        help='number of iterations to display',
                        default=10000, type=int)

    parser.add_argument('--model_dir', dest='model_dir',
                        help='directory to save models', default="models",
                        type=str)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=4, type=int)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=3, type=int)
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--gamma_altitude', dest='gamma_altitude',
                        help='the gamma is used to control the relative weight of the adversarial loss of altitude',
                        type=float, required=True)
    parser.add_argument('--gamma_angle', dest='gamma_angle',
                        help='the gamma is used to control the relative weight of the adversarial loss of viewing angle',
                        type=float, required=True)
    parser.add_argument('--gamma_weather', dest='gamma_weather',
                        help='the gamma is used to control the relative weight of the adversarial loss of weather',
                        type=float, required=True)
    # config optimization
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="adam", type=str)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.001, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=5, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)

    # set training session
    parser.add_argument('--s', dest='session',
                        help='training session',
                        default=1, type=int)

    # resume trained model
    parser.add_argument('--r', dest='resume',
                        help='resume checkpoint or not',
                        default=False, type=str2bool)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load model',
                        default=4, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load model',
                        default=3960, type=int)
    # log and diaplay
    parser.add_argument('--use_tfb', dest='use_tfboard',
                        help='whether use tensorboard',
                        action='store_true')

    parser.add_argument('--n_minibatches', dest='n_minibatches',
                        help='number of minibatches',
                        default=4, type=int)
    parser.add_argument('--n_minibatches_eval', dest='n_minibatches_eval',
                        help='number of minibatches for evaluation',
                        default=16, type=int)
    parser.add_argument('--use_restarting', dest='use_restarting',
                        help='where to use restarting',
                        action='store_true')
    parser.add_argument('--restarting_iters', dest='restarting_iters',
                        help='number of steps for restarting',
                        default=1000, type=int)
    parser.add_argument('--retraining_steps', dest='retraining_steps',
                        help='number of steps for retraining',
                        default=250, type=int)
    parser.add_argument('--monitor_discriminator', dest='monitor_discriminator',
                        help='whether to monitor the accuracy of the discriminator',
                        action='store_true')
    parser.add_argument('--use_adversarial_loss', dest='use_adversarial_loss',
                        help='whether to use adversarial training',
                        action='store_true')
    parser.add_argument('--save_iters', dest='save_iters',
                        help='number of iterations for saving the model',
                        default=1000, type=int)
    parser.add_argument('--summary_dir', dest='summary_dir',
                        help='the directory of the summary txt files', default='summaries',
                        type=str)
    parser.add_argument('--angle_thresh', dest='angle_thresh',
                        help='the threshold of the discriminator monitor', default=0.9,
                        type=float)
    parser.add_argument('--altitude_thresh', dest='altitude_thresh',
                        help='the threshold of the discriminator monitor', default=0.9,
                        type=float)
    parser.add_argument('--weather_thresh', dest='weather_thresh',
                        help='the threshold of the discriminator monitor', default=0.9,
                        type=float)
    parser.add_argument('--eval_display', dest='eval_display',
                        help='display the evaluation results regularly',
                        action='store_true')
    parser.add_argument('--niter', dest='niter',
                        help='number of iteration as starting',
                        default=0, type=int)

    args = parser.parse_args()
    return args

def exponential_decay(gamma0, alpha, x):
    return gamma0 - gamma0 * math.pow(1 - alpha, x)

def polynomial_decay(gamma0, beta, x, x_t):
    return gamma0 - gamma0 * math.pow(1 - x / x_t, beta)

class sampler(Sampler):
    def __init__(self, train_size, batch_size):
        self.num_data = train_size
        self.num_per_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0, batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if train_size % batch_size:
            self.leftover = torch.arange(self.num_per_batch * batch_size, train_size).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(self.num_per_batch).view(-1, 1) * self.batch_size
        self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover), 0)

        return iter(self.rand_num_view)

    def __len__(self):
        return self.num_data


if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)

    assert args.dataset == "uav"
    args.imdb_name = "uav_2017_trainval"
    args.imdbval_name = "uav_2017_test"
    args.set_cfgs = ['ANCHOR_SCALES', '[1, 2, 4, 8, 16]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '107']

    args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    # torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    cfg.TRAIN.USE_FLIPPED = False
    cfg.USE_GPU_NMS = args.cuda

    '''Dataloader for the training'''
    imdb_train, roidb_train, ratio_list_train, ratio_index_train = combined_roidb(args.imdb_name)
    train_size = len(roidb_train)
    print('{:d} roidb entries'.format(len(roidb_train)))
    sampler_batch = sampler(train_size, args.batch_size)
    dataset_train = roibatchLoader(roidb_train, ratio_list_train, ratio_index_train, args.batch_size, \
                             imdb_train.num_classes, training=True)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,
                                             sampler=sampler_batch, num_workers=args.num_workers)

    '''Dataloader for the validation/testing'''
    imdb_val, roidb_val, ratio_list_val, ratio_index_val = combined_roidb(args.imdbval_name)
    val_size = len(roidb_val)
    print('{:d} roidb entries'.format(len(roidb_train)))
    sampler_batch = sampler(val_size, args.batch_size)
    dataset_val = roibatchLoader(roidb_val, ratio_list_val, ratio_index_val, args.batch_size, \
                             imdb_val.num_classes, training=True)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size,
                                             sampler=sampler_batch, num_workers=args.num_workers)

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    meta_data = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        meta_data = meta_data.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    meta_data = Variable(meta_data)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

    if args.cuda:
        cfg.CUDA = True

    # initilize the network here.
    if args.net == 'vgg16':
        fasterRCNN = vgg16(imdb_train.classes, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(imdb_train.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fasterRCNN = resnet(imdb_train.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        fasterRCNN = resnet(imdb_train.classes, 152, pretrained=True, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    lr = cfg.TRAIN.LEARNING_RATE
    lr = args.lr
    # tr_momentum = cfg.TRAIN.MOMENTUM
    # tr_momentum = args.momentum

    params_util = []
    params_adv = []
    params_aux = []
    params_keys_util = []
    params_keys_adv = []
    params_keys_aux = []

    for key, value in dict(fasterRCNN.named_parameters()).items():
        #print(key)
        if value.requires_grad and not any(name in key for name in ['RCNN_weather', 'RCNN_altitude', 'RCNN_angle',
                                                                    'RCNN_weather_score', 'RCNN_altitude_score', 'RCNN_angle_score']):
            params_keys_util.append(key)
            if 'bias' in key:
                params_util += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                                 'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params_util += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
        if value.requires_grad and any(name in key for name in ['RCNN_weather', 'RCNN_altitude', 'RCNN_angle',
                                                                'RCNN_weather_score', 'RCNN_altitude_score', 'RCNN_angle_score']):
            params_keys_aux.append(key)
            if 'bias' in key:
                params_aux += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params_aux += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
        if value.requires_grad and any(name in key for name in ['RCNN_base']):
            params_keys_adv.append(key)
            if 'bias' in key:
                params_adv += [{'params': [value], 'lr': lr * 0.1 * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params_adv += [{'params': [value], 'lr': lr * 0.1, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params_util)

    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_util, momentum=cfg.TRAIN.MOMENTUM)

    # print(params_keys_util)
    # print(params_keys_aux)
    # print(params_keys_adv)
    aux_optimizer = torch.optim.Adam(params_aux)
    # adv_optimizer = torch.optim.Adam(params_adv)

    if args.gamma_altitude > 1e-10 and args.gamma_angle > 1e-10 and args.gamma_weather > 1e-10:
        nuisance_type = "A+V+W"
    elif args.gamma_altitude > 1e-10 and args.gamma_angle > 1e-10:
        nuisance_type = "A+V"
    elif args.gamma_altitude > 1e-10 and args.gamma_weather > 1e-10:
        nuisance_type = "A+W"
    elif args.gamma_altitude > 1e-10:
        nuisance_type = "A"
    elif args.gamma_angle > 1e-10:
        nuisance_type = "V"
    elif args.gamma_weather > 1e-10:
        nuisance_type = "W"
    else:
        nuisance_type ="Baseline"

    model_dir = os.path.join(args.model_dir, nuisance_type, 'altitude={}_angle={}_weather={}'.format(str(args.gamma_altitude), str(args.gamma_angle), str(args.gamma_weather)))
    summary_dir = os.path.join(args.summary_dir, nuisance_type, 'altitude={}_angle={}_weather={}'.format(str(args.gamma_altitude), str(args.gamma_angle), str(args.gamma_weather)))



    if args.resume:
        load_name = os.path.join(model_dir, 'faster_rcnn_{}_{}_{}_adv.pth'.format(args.checksession, args.checkepoch,
                                                                       args.checkpoint))
        print("loading checkpoint %s" % (load_name))

        checkpoint = torch.load(load_name)
        args.session = checkpoint['session']
        args.start_epoch = checkpoint['epoch']

        model_dict = fasterRCNN.state_dict()
        model_dict.update(checkpoint['model'])
        fasterRCNN.load_state_dict(model_dict)

        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k,v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        aux_optimizer.load_state_dict(checkpoint['aux_optimizer'])
        for state in aux_optimizer.state.values():
            for k,v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        print("loaded checkpoint %s" % (load_name))
    else:
        load_name = os.path.join(os.path.join(args.model_dir, 'Pretrained'),
                                 'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
        print("loading checkpoint %s" % (load_name))

        checkpoint = torch.load(load_name)
        args.session = checkpoint['session']
        args.start_epoch = checkpoint['epoch']

        model_dict = fasterRCNN.state_dict()
        model_dict.update(checkpoint['model'])
        fasterRCNN.load_state_dict(model_dict)

        print("loaded checkpoint %s" % (load_name))

    if args.mGPUs:
        fasterRCNN = nn.DataParallel(fasterRCNN)

    if args.cuda:
        fasterRCNN.cuda()

    iters_per_epoch = int(train_size / args.batch_size)

    if args.use_tfboard:
        from tensorboardX import SummaryWriter

        logger = SummaryWriter("logs")

    niter = args.niter
    fasterRCNN.train()
    data_iter_train = iter(dataloader_train)
    data_iter_val = iter(dataloader_val)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    train_summary_file = open(os.path.join(summary_dir, 'train_summary.txt'), 'w', 0)
    val_summary_file = open(os.path.join(summary_dir, 'val_summary.txt'), 'w', 0)

    while niter < iters_per_epoch * ((args.max_epochs - args.start_epoch) + 1):
        fasterRCNN.zero_grad()
        # setting to train mode
        start = time.time()

        if niter % ((args.lr_decay_step + 1) * iters_per_epoch) == 0:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma

        if niter == 0 or (args.use_restarting and niter % args.restarting_iters == 0):
            '''
            Restarting
            '''
            caffe_state_dict = torch.load('data/pretrained_model/resnet101_caffe.pth')
            rcnn_altitude = fasterRCNN.module.RCNN_altitude
            rcnn_altitude.load_state_dict({'0'+k[6:]: v for k, v in caffe_state_dict.items() if ('layer4' in k) and ('0'+k[6:] in rcnn_altitude.state_dict())})
            rcnn_angle = fasterRCNN.module.RCNN_angle
            rcnn_angle.load_state_dict({'0'+k[6:]: v for k, v in caffe_state_dict.items() if ('layer4' in k) and ('0'+k[6:] in rcnn_angle.state_dict())})
            rcnn_weather = fasterRCNN.module.RCNN_weather
            rcnn_weather.load_state_dict({'0'+k[6:]: v for k, v in caffe_state_dict.items() if ('layer4' in k) and ('0'+k[6:] in rcnn_weather.state_dict())})

            for _ in itertools.repeat(None, args.retraining_steps):
                loss_adv_temp = 0
                loss_aux_temp = 0
                acc_altitude_temp = 0
                acc_angle_temp = 0
                acc_weather_temp = 0
                aux_optimizer.zero_grad()
                for _ in itertools.repeat(None, args.n_minibatches):
                    try:
                        data = next(data_iter_train)
                    except StopIteration:
                        data_iter_train = iter(dataloader_train)
                        data = next(data_iter_train)
                    im_data.data.resize_(data[0].size()).copy_(data[0])
                    im_info.data.resize_(data[1].size()).copy_(data[1])
                    meta_data.data.resize_(data[2].size()).copy_(data[2])
                    gt_boxes.data.resize_(data[3].size()).copy_(data[3])
                    num_boxes.data.resize_(data[4].size()).copy_(data[4])

                    RCNN_loss_altitude, RCNN_loss_altitude_adv, RCNN_acc_altitude, \
                    RCNN_loss_angle, RCNN_loss_angle_adv, RCNN_acc_angle, \
                    RCNN_loss_weather, RCNN_loss_weather_adv, RCNN_acc_weather = fasterRCNN(im_data, im_info, meta_data, gt_boxes, num_boxes, run_partial=True)

                    loss_altitude = RCNN_loss_altitude.mean()
                    loss_altitude_adv = RCNN_loss_altitude_adv.mean()
                    acc_altitude = RCNN_acc_altitude.mean()
                    loss_angle = RCNN_loss_angle.mean()
                    loss_angle_adv = RCNN_loss_angle_adv.mean()
                    acc_angle = RCNN_acc_angle.mean()
                    loss_weather = RCNN_loss_weather.mean()
                    loss_weather_adv = RCNN_loss_weather_adv.mean()
                    acc_weather = RCNN_acc_weather.mean()

                    acc_altitude = acc_altitude / args.n_minibatches
                    loss_altitude_adv = loss_altitude_adv / args.n_minibatches
                    loss_altitude_aux = loss_altitude / args.n_minibatches
                    acc_angle = acc_angle /args.n_minibatches
                    loss_angle_adv = loss_angle_adv / args.n_minibatches
                    loss_angle_aux = loss_angle / args.n_minibatches
                    acc_weather = acc_weather / args.n_minibatches
                    loss_weather_adv = loss_weather_adv / args.n_minibatches
                    loss_weather_aux = loss_weather / args.n_minibatches

                    loss_adv = loss_altitude_adv + loss_angle_adv + loss_weather_adv
                    loss_aux = loss_altitude_aux + loss_angle_aux + loss_weather_aux

                    loss_adv_temp += loss_adv.item()
                    loss_aux_temp += loss_aux.item()
                    acc_altitude_temp += acc_altitude.item()
                    acc_angle_temp += acc_angle.item()
                    acc_weather_temp += acc_weather.item()

                    loss_aux.backward(retain_graph=False)
                aux_optimizer.step()
                if niter == 0:
                    print(
                        "Initialization (Auxiliary): [session %d][epoch %2d][iter %4d/%4d] altitude accuracy: %.4f, angle accuracy: %.4f, weather accuracy: %.4f, auxiliary loss: %.4f, adversarial loss: %.4f, lr: %.2e" \
                        % (args.session, niter // iters_per_epoch + 1, niter % iters_per_epoch, iters_per_epoch,
                           acc_altitude_temp, acc_angle_temp, acc_weather_temp, loss_aux_temp, loss_adv_temp, lr))
                else:
                    print(
                        "Restarting (Auxiliary): [session %d][epoch %2d][iter %4d/%4d] altitude accuracy: %.4f, angle accuracy: %.4f, weather accuracy: %.4f, auxiliary loss: %.4f, adversarial loss: %.4f, lr: %.2e" \
                        % (args.session, niter // iters_per_epoch + 1, niter % iters_per_epoch, iters_per_epoch,
                           acc_altitude_temp, acc_angle_temp, acc_weather_temp, loss_aux_temp, loss_adv_temp, lr))

        if args.use_adversarial_loss:
            loss_temp = 0
            loss_util_temp = 0
            loss_adv_temp = 0
            loss_aux_temp = 0
            optimizer.zero_grad()
            for _ in itertools.repeat(None, args.n_minibatches):
                try:
                    data = next(data_iter_train)
                except StopIteration:
                    data_iter_train = iter(dataloader_train)
                    data = next(data_iter_train)
                im_data.data.resize_(data[0].size()).copy_(data[0])
                im_info.data.resize_(data[1].size()).copy_(data[1])
                meta_data.data.resize_(data[2].size()).copy_(data[2])
                gt_boxes.data.resize_(data[3].size()).copy_(data[3])
                num_boxes.data.resize_(data[4].size()).copy_(data[4])

                rois, cls_prob, bbox_pred, \
                rpn_loss_cls, rpn_loss_box, \
                RCNN_loss_cls, RCNN_loss_bbox, \
                RCNN_loss_altitude, RCNN_loss_altitude_adv, RCNN_acc_altitude, \
                RCNN_loss_angle, RCNN_loss_angle_adv, RCNN_acc_angle, \
                RCNN_loss_weather, RCNN_loss_weather_adv, RCNN_acc_weather, \
                rois_label = fasterRCNN(im_data, im_info, meta_data, gt_boxes, num_boxes, run_partial=False)

                loss_rpn = rpn_loss_cls.mean() + rpn_loss_box.mean()
                loss_rcnn = RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()

                loss_altitude = RCNN_loss_altitude.mean()
                loss_altitude_adv = RCNN_loss_altitude_adv.mean()
                loss_angle = RCNN_loss_angle.mean()
                loss_angle_adv = RCNN_loss_angle_adv.mean()
                loss_weather = RCNN_loss_weather.mean()
                loss_weather_adv = RCNN_loss_weather_adv.mean()

                loss_util = (loss_rpn + loss_rcnn)  / args.n_minibatches
                loss_altitude_adv = loss_altitude_adv  / args.n_minibatches
                loss_altitude_aux = loss_altitude / args.n_minibatches
                loss_angle_adv = loss_angle_adv / args.n_minibatches
                loss_angle_aux = loss_angle / args.n_minibatches
                loss_weather_adv = loss_weather_adv / args.n_minibatches
                loss_weather_aux = loss_weather / args.n_minibatches

                loss_adv = loss_altitude_adv + loss_angle_adv + loss_weather_adv
                loss_aux = loss_altitude_aux + loss_angle_aux + loss_weather_aux
                loss = loss_util + args.gamma_altitude * loss_altitude_adv + args.gamma_angle * loss_angle_adv + args.gamma_weather * loss_weather_adv

                loss_util_temp += loss_util.item()
                loss_adv_temp += loss_adv.item()
                loss_aux_temp += loss_aux.item()
                loss_temp += loss.item()

                loss.backward(retain_graph=False)
            optimizer.step()
            print(
                "Alternating Training (Utility + Adversarial): [session %d][epoch %2d][iter %4d/%4d] utility loss: %.4f, auxiliary loss: %.4f, adversarial loss: %.4f, utility+adversarial loss: %.4f, lr: %.2e" \
                % (args.session, niter // iters_per_epoch + 1, niter % iters_per_epoch, iters_per_epoch,
                    loss_util_temp, loss_aux_temp, loss_adv_temp, loss_temp, lr))

        if args.monitor_discriminator:
            while True:
                loss_adv_temp = 0
                loss_aux_temp = 0
                acc_altitude_temp = 0
                acc_angle_temp = 0
                acc_weather_temp = 0
                aux_optimizer.zero_grad()
                for _ in itertools.repeat(None, args.n_minibatches):
                    try:
                        data = next(data_iter_train)
                    except StopIteration:
                        data_iter_train = iter(dataloader_train)
                        data = next(data_iter_train)
                    im_data.data.resize_(data[0].size()).copy_(data[0])
                    im_info.data.resize_(data[1].size()).copy_(data[1])
                    meta_data.data.resize_(data[2].size()).copy_(data[2])
                    gt_boxes.data.resize_(data[3].size()).copy_(data[3])
                    num_boxes.data.resize_(data[4].size()).copy_(data[4])

                    RCNN_loss_altitude, RCNN_loss_altitude_adv, RCNN_acc_altitude, \
                    RCNN_loss_angle, RCNN_loss_angle_adv, RCNN_acc_angle, \
                    RCNN_loss_weather, RCNN_loss_weather_adv, RCNN_acc_weather = fasterRCNN(im_data, im_info, meta_data, gt_boxes, num_boxes, run_partial=True)

                    loss_altitude = RCNN_loss_altitude.mean()
                    loss_altitude_adv = RCNN_loss_altitude_adv.mean()
                    acc_altitude = RCNN_acc_altitude.mean()
                    loss_angle = RCNN_loss_angle.mean()
                    loss_angle_adv = RCNN_loss_angle_adv.mean()
                    acc_angle = RCNN_acc_angle.mean()
                    loss_weather = RCNN_loss_weather.mean()
                    loss_weather_adv = RCNN_loss_weather_adv.mean()
                    acc_weather = RCNN_acc_weather.mean()

                    acc_altitude = acc_altitude / args.n_minibatches
                    loss_altitude_adv = loss_altitude_adv / args.n_minibatches
                    loss_altitude_aux = loss_altitude / args.n_minibatches
                    acc_angle = acc_angle / args.n_minibatches
                    loss_angle_adv = loss_angle_adv / args.n_minibatches
                    loss_angle_aux = loss_angle / args.n_minibatches
                    acc_weather = acc_weather / args.n_minibatches
                    loss_weather_adv = loss_weather_adv / args.n_minibatches
                    loss_weather_aux = loss_weather / args.n_minibatches

                    loss_adv = loss_altitude_adv + loss_angle_adv + loss_weather_adv
                    loss_aux = loss_altitude_aux + loss_angle_aux + loss_weather_aux

                    loss_adv_temp += loss_adv.item()
                    loss_aux_temp += loss_aux.item()
                    acc_altitude_temp += acc_altitude.item()
                    acc_angle_temp += acc_angle.item()
                    acc_weather_temp += acc_weather.item()

                    loss_aux.backward(retain_graph=False)
                aux_optimizer.step()
                print(
                    "Alternating Training (Auxiliary): [session %d][epoch %2d][iter %4d/%4d] altitude accuracy: %.4f, angle accuracy: %.4f, weather accuracy: %.4f, auxiliary loss: %.4f, adversarial loss: %.4f, lr: %.2e" \
                    % (args.session, niter // iters_per_epoch + 1, niter % iters_per_epoch, iters_per_epoch,
                        acc_altitude_temp, acc_angle_temp, acc_weather_temp, loss_aux_temp, loss_adv_temp, lr))

                if acc_angle_temp > args.angle_thresh and acc_altitude_temp > args.altitude_thresh and acc_weather_temp > args.weather_thresh:
                    break

        if args.eval_display:
            if niter % args.disp_interval == 0:
                end = time.time()

                '''Training evaluation'''
                loss_temp = 0
                loss_util_temp = 0
                loss_adv_temp = 0
                loss_aux_temp = 0
                acc_altitude_temp = 0
                acc_angle_temp = 0
                acc_weather_temp = 0

                loss_rpn_cls_temp = 0
                loss_rpn_box_temp = 0
                loss_rcnn_cls_temp = 0
                loss_rcnn_box_temp = 0
                fg_cnt = 0
                bg_cnt = 0
                with torch.no_grad():
                    for _ in itertools.repeat(None, args.n_minibatches_eval):
                        try:
                            data = next(data_iter_train)
                        except StopIteration:
                            data_iter_train = iter(dataloader_train)
                            data = next(data_iter_train)
                        im_data.data.resize_(data[0].size()).copy_(data[0])
                        im_info.data.resize_(data[1].size()).copy_(data[1])
                        meta_data.data.resize_(data[2].size()).copy_(data[2])
                        gt_boxes.data.resize_(data[3].size()).copy_(data[3])
                        num_boxes.data.resize_(data[4].size()).copy_(data[4])

                        rois, cls_prob, bbox_pred, \
                        rpn_loss_cls, rpn_loss_box, \
                        RCNN_loss_cls, RCNN_loss_bbox, \
                        RCNN_loss_altitude, RCNN_loss_altitude_adv, RCNN_acc_altitude, \
                        RCNN_loss_angle, RCNN_loss_angle_adv, RCNN_acc_angle, \
                        RCNN_loss_weather, RCNN_loss_weather_adv, RCNN_acc_weather, \
                        rois_label = fasterRCNN(im_data, im_info, meta_data, gt_boxes, num_boxes, run_partial=False)

                        loss_rpn = rpn_loss_cls.mean() + rpn_loss_box.mean()
                        loss_rcnn = RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
                        loss_altitude = RCNN_loss_altitude.mean()
                        loss_altitude_adv = RCNN_loss_altitude_adv.mean()
                        acc_altitude = RCNN_acc_altitude.mean()
                        loss_angle = RCNN_loss_angle.mean()
                        loss_angle_adv = RCNN_loss_angle_adv.mean()
                        acc_angle = RCNN_acc_angle.mean()
                        loss_weather = RCNN_loss_weather.mean()
                        loss_weather_adv = RCNN_loss_weather_adv.mean()
                        acc_weather = RCNN_acc_weather.mean()

                        rpn_loss_cls = rpn_loss_cls.mean()
                        rpn_loss_box = rpn_loss_box.mean()
                        RCNN_loss_cls = RCNN_loss_cls.mean()
                        RCNN_loss_bbox = RCNN_loss_bbox.mean()

                        loss_rpn_cls = rpn_loss_cls / args.n_minibatches_eval
                        loss_rpn_box = rpn_loss_box / args.n_minibatches_eval
                        loss_rcnn_cls = RCNN_loss_cls / args.n_minibatches_eval
                        loss_rcnn_box = RCNN_loss_bbox / args.n_minibatches_eval
                        loss_util = (loss_rpn + loss_rcnn) / args.n_minibatches_eval
                        loss_altitude_adv = loss_altitude_adv / args.n_minibatches_eval
                        loss_altitude_aux = loss_altitude / args.n_minibatches_eval
                        acc_altitude = acc_altitude / args.n_minibatches_eval
                        loss_angle_adv = loss_angle_adv / args.n_minibatches_eval
                        loss_angle_aux = loss_angle / args.n_minibatches_eval
                        acc_angle = acc_angle / args.n_minibatches_eval
                        loss_weather_adv = loss_weather_adv / args.n_minibatches_eval
                        loss_weather_aux = loss_weather / args.n_minibatches_eval
                        acc_weather = acc_weather / args.n_minibatches_eval

                        loss = loss_util + args.gamma_altitude * loss_altitude_adv + args.gamma_angle * loss_angle_adv + args.gamma_weather * loss_weather_adv
                        loss_adv = loss_altitude_adv + loss_angle_adv + loss_weather_adv
                        loss_aux = loss_altitude_aux + loss_angle_aux + loss_weather_aux

                        loss_util_temp += loss_util.item()
                        loss_adv_temp += loss_adv.item()
                        loss_aux_temp += loss_aux.item()
                        loss_temp += loss.item()
                        acc_altitude_temp += acc_altitude.item()
                        acc_angle_temp += acc_angle.item()
                        acc_weather_temp += acc_weather.item()

                        loss_rpn_cls_temp += loss_rpn_cls
                        loss_rpn_box_temp += loss_rpn_box
                        loss_rcnn_cls_temp += loss_rcnn_cls
                        loss_rcnn_box_temp += loss_rcnn_box

                        fg_cnt += torch.sum(rois_label.data.ne(0))
                        bg_cnt += (rois_label.data.numel() - fg_cnt)
                print(
                    "**********DISPLAY TRAINING**********: [session %d][epoch %2d][iter %4d/%4d] altitude accuracy: %.4f, angle accuracy: %.4f, weather accuracy: %.4f, utility loss: %.4f, auxiliary loss: %.4f, adversarial loss: %.4f, utility+adversarial loss: %.4f, lr: %.2e" \
                    % (args.session, niter // iters_per_epoch + 1, niter % iters_per_epoch, iters_per_epoch,
                       acc_altitude_temp, acc_angle_temp, acc_weather_temp, loss_util_temp, loss_aux_temp, loss_adv_temp, loss_temp, lr))
                print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end - start))
                print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                    % (loss_rpn_cls_temp, loss_rpn_box_temp, loss_rcnn_cls_temp, loss_rcnn_box_temp))
                if args.use_tfboard:
                    info = {
                        'training_altitude_acc': acc_altitude_temp,
                        'training_angle_acc': acc_angle_temp,
                        'training_weather_acc': acc_weather_temp,
                        'train_loss': loss_temp,
                        'train_loss_util': loss_util_temp,
                        'train_loss_aux': loss_aux_temp,
                        'train_loss_adv': loss_adv_temp,
                        'train_loss_rpn_cls': loss_rpn_cls_temp,
                        'train_loss_rpn_box': loss_rpn_box_temp,
                        'train_loss_rcnn_cls': loss_rcnn_cls_temp,
                        'train_loss_rcnn_box': loss_rcnn_box_temp
                    }
                    logger.add_scalars(
                        "logs_altitude={}_angle={}_weather=_{}/losses_train".format(args.gamma_altitude, args.gamma_angle, args.gamma_weather), info, niter)

                train_summary_file.write("[session %d][epoch %2d][iter %4d/%4d] altitude accuracy: %.4f, angle accuracy: %.4f, weather accuracy: %.4f, utility loss: %.4f, auxiliary loss: %.4f, adversarial loss: %.4f, utility+adversarial loss: %.4f, lr: %.2e\n" \
                    % (args.session, niter // iters_per_epoch + 1, niter % iters_per_epoch, iters_per_epoch,
                       acc_altitude_temp, acc_angle_temp, acc_weather_temp, loss_util_temp, loss_aux_temp, loss_adv_temp, loss_temp, lr))
                train_summary_file.write("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f\n" \
                    % (loss_rpn_cls_temp, loss_rpn_box_temp, loss_rcnn_cls_temp, loss_rcnn_box_temp))

                '''Validation evaluation'''
                loss_temp = 0
                loss_util_temp = 0
                loss_adv_temp = 0
                loss_aux_temp = 0
                acc_altitude_temp = 0
                acc_angle_temp = 0
                acc_weather_temp = 0

                loss_rpn_cls_temp = 0
                loss_rpn_box_temp = 0
                loss_rcnn_cls_temp = 0
                loss_rcnn_box_temp = 0
                fg_cnt = 0
                bg_cnt = 0
                with torch.no_grad():
                    for _ in itertools.repeat(None, args.n_minibatches_eval):
                        try:
                            data = next(data_iter_val)
                        except StopIteration:
                            data_iter_val = iter(dataloader_val)
                            data = next(data_iter_val)
                        im_data.data.resize_(data[0].size()).copy_(data[0])
                        im_info.data.resize_(data[1].size()).copy_(data[1])
                        meta_data.data.resize_(data[2].size()).copy_(data[2])
                        gt_boxes.data.resize_(data[3].size()).copy_(data[3])
                        num_boxes.data.resize_(data[4].size()).copy_(data[4])

                        rois, cls_prob, bbox_pred, \
                        rpn_loss_cls, rpn_loss_box, \
                        RCNN_loss_cls, RCNN_loss_bbox, \
                        RCNN_loss_altitude, RCNN_loss_altitude_adv, RCNN_acc_altitude, \
                        RCNN_loss_angle, RCNN_loss_angle_adv, RCNN_acc_angle, \
                        RCNN_loss_weather, RCNN_loss_weather_adv, RCNN_acc_weather, \
                        rois_label = fasterRCNN(im_data, im_info, meta_data, gt_boxes, num_boxes, run_partial=False)

                        loss_rpn = rpn_loss_cls.mean() + rpn_loss_box.mean()
                        loss_rcnn = RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
                        loss_altitude = RCNN_loss_altitude.mean()
                        loss_altitude_adv = RCNN_loss_altitude_adv.mean()
                        acc_altitude = RCNN_acc_altitude.mean()
                        loss_angle = RCNN_loss_angle.mean()
                        loss_angle_adv = RCNN_loss_angle_adv.mean()
                        acc_angle = RCNN_acc_angle.mean()
                        loss_weather = RCNN_loss_weather.mean()
                        loss_weather_adv = RCNN_loss_weather_adv.mean()
                        acc_weather = RCNN_acc_weather.mean()
                        rpn_loss_cls = rpn_loss_cls.mean()
                        rpn_loss_box = rpn_loss_box.mean()
                        RCNN_loss_cls = RCNN_loss_cls.mean()
                        RCNN_loss_bbox = RCNN_loss_bbox.mean()

                        loss_rpn_cls = rpn_loss_cls / args.n_minibatches_eval
                        loss_rpn_box = rpn_loss_box / args.n_minibatches_eval
                        loss_rcnn_cls = RCNN_loss_cls / args.n_minibatches_eval
                        loss_rcnn_box = RCNN_loss_bbox / args.n_minibatches_eval
                        loss_util = (loss_rpn + loss_rcnn) / args.n_minibatches_eval
                        loss_altitude_adv = loss_altitude_adv / args.n_minibatches_eval
                        loss_altitude_aux = loss_altitude / args.n_minibatches_eval
                        acc_altitude = acc_altitude / args.n_minibatches_eval
                        loss_angle_adv = loss_angle_adv / args.n_minibatches_eval
                        loss_angle_aux = loss_angle / args.n_minibatches_eval
                        acc_angle = acc_angle / args.n_minibatches_eval
                        loss_weather_adv = loss_weather_adv / args.n_minibatches_eval
                        loss_weather_aux = loss_weather / args.n_minibatches_eval
                        acc_weather = acc_weather / args.n_minibatches_eval

                        loss = loss_util + args.gamma_altitude * loss_altitude_adv + args.gamma_angle * loss_angle_adv + args.gamma_weather * loss_weather_adv
                        loss_adv = loss_altitude_adv + loss_angle_adv + loss_weather_adv
                        loss_aux = loss_altitude_aux + loss_angle_aux + loss_weather_aux

                        loss_util_temp += loss_util.item()
                        loss_adv_temp += loss_adv.item()
                        loss_aux_temp += loss_aux.item()
                        loss_temp += loss.item()
                        acc_altitude_temp += acc_altitude.item()
                        acc_angle_temp += acc_angle.item()
                        acc_weather_temp += acc_weather.item()

                        loss_rpn_cls_temp += loss_rpn_cls
                        loss_rpn_box_temp += loss_rpn_box
                        loss_rcnn_cls_temp += loss_rcnn_cls
                        loss_rcnn_box_temp += loss_rcnn_box

                        fg_cnt += torch.sum(rois_label.data.ne(0))
                        bg_cnt += (rois_label.data.numel() - fg_cnt)
                print(
                    "**********DISPLAY VALIDATION**********: [session %d][epoch %2d][iter %4d/%4d] altitude accuracy: %.4f, angle accuracy: %.4f, weather accuracy: %.4f, utility loss: %.4f, auxiliary loss: %.4f, adversarial loss: %.4f, utility+adversarial loss: %.4f, lr: %.2e" \
                    % (args.session, niter // iters_per_epoch + 1, niter % iters_per_epoch, iters_per_epoch,
                       acc_altitude_temp, acc_angle_temp, acc_weather_temp, loss_util_temp, loss_aux_temp, loss_adv_temp, loss_temp, lr))
                print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end - start))
                print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                      % (loss_rpn_cls_temp, loss_rpn_box_temp, loss_rcnn_cls_temp, loss_rcnn_box_temp))
                if args.use_tfboard:
                    info = {
                        'val_altitude_acc': acc_altitude_temp,
                        'val_angle_acc': acc_angle_temp,
                        'val_weather_acc': acc_weather_temp,
                        'val_loss': loss_temp,
                        'val_loss_util': loss_util_temp,
                        'val_loss_aux': loss_aux_temp,
                        'val_loss_adv': loss_adv_temp,
                        'val_loss_rpn_cls': loss_rpn_cls_temp,
                        'val_loss_rpn_box': loss_rpn_box_temp,
                        'val_loss_rcnn_cls': loss_rcnn_cls_temp,
                        'val_loss_rcnn_box': loss_rcnn_box_temp
                    }
                    logger.add_scalars("logs_altitude={}_angle={}_weather={}_{}/losses_val".format(args.gamma_altitude, args.gamma_angle, args.gamma_weather), info, niter)

                val_summary_file.write("[session %d][epoch %2d][iter %4d/%4d] altitude accuracy: %.4f, angle accuracy: %.4f, weather_accuracy: %.4f, utility loss: %.4f, auxiliary loss: %.4f, adversarial loss: %.4f, utility+adversarial loss: %.4f, lr: %.2e\n" \
                        % (args.session, niter // iters_per_epoch + 1, niter % iters_per_epoch, iters_per_epoch,
                        acc_altitude_temp, acc_angle_temp, acc_weather_temp, loss_util_temp, loss_aux_temp, loss_adv_temp, loss_temp, lr))
                val_summary_file.write("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f\n" \
                        % (loss_rpn_cls_temp, loss_rpn_box_temp, loss_rcnn_cls_temp, loss_rcnn_box_temp))
                start = time.time()

        if niter % args.save_iters == 0:
            save_name = os.path.join(model_dir, 'faster_rcnn_{}_{}_{}_adv.pth'.format(args.session,
                                                                                          niter // iters_per_epoch + 1,
                                                                                          niter % iters_per_epoch))
            save_checkpoint({
                'session': args.session,
                'epoch': niter // iters_per_epoch + 1,
                'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
                'optimizer': optimizer.state_dict(),
                'aux_optimizer': aux_optimizer.state_dict(),
                'pooling_mode': cfg.POOLING_MODE,
                'class_agnostic': args.class_agnostic,
            }, save_name)
            print('save model: {}'.format(save_name))

        niter += 1

    train_summary_file.close()
    val_summary_file.close()

    if args.use_tfboard:
        logger.close()
