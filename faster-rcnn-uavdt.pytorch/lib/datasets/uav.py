from __future__ import print_function
from __future__ import absolute_import
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import xml.dom.minidom as minidom

import os
# import PIL
import numpy as np
import scipy.sparse
import subprocess
import math
import glob
import uuid
import scipy.io as sio
import xml.etree.ElementTree as ET
try:
    import cPickle
except ImportError:
    import _pickle as cPickle
from .imdb import imdb
from .imdb import ROOT_DIR
from . import ds_utils
from .voc_eval import voc_eval

# TODO: make fast_rcnn irrelevant
# >>>> obsolete, because it depends on sth outside of this project
from model.utils.config import cfg

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

# <<<< obsolete


class uav(imdb):
    def __init__(self, image_set, year, devkit_path=None):
        imdb.__init__(self, 'uav_' + year + '_' + image_set)
        self._year = year
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
        self._data_path = os.path.join(self._devkit_path, 'UAV' + self._year)
        self._classes = ('__background__', 'car')
        self._weathers = ('daylight', 'night')
        self._altitudes = ('low-alt', 'medium-alt', 'high-alt')
        self._angles = ('front-view', 'side-view', 'bird-view')
        #self._angles = ('front-side-view', 'front-view', 'side-view', 'bird-view')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._weather_to_ind = {'daylight':0, 'night':1}
        self._altitude_to_ind = {'low-alt':0, 'medium-alt':1, 'high-alt':2}
        self._angle_to_ind = {'front-view': 0, 'side-view': 1, 'bird-view': 2}
        #self._angle_to_ind = {'front-side-view': 0, 'front-view': 0, 'side-view': 0, 'bird-view': 1}
        #self._angle_to_ind = {'front-side-view':0, 'front-view':1, 'side-view':2, 'bird-view':3}
        #self._angle_to_ind = {'front-side-view': 0, 'front-view': 1, 'side-view': 0, 'bird-view': 2}
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'
        # PASCAL specific config options
        self.config = {'cleanup'     : True,
                       'use_salt'    : True,
                       'use_diff'    : False,
                       'matlab_eval' : False,
                       'rpn_file'    : None,
                       'min_size'    : 4}

        assert os.path.exists(self._devkit_path), \
                'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)
        self._gamma_altitude = None
        self._gamma_angle = None
        self._gamma_weather = None
        self._epoch = None
        self._ckpt = None

    def set_gamma_altitude(self, gamma):
        self._gamma_altitude = gamma

    def set_gamma_angle(self, gamma):
        self._gamma_angle = gamma

    def set_gamma_weather(self, gamma):
        self._gamma_weather = gamma

    def set_epoch(self, epoch):
        self._epoch = epoch

    def set_ckpt(self, ckpt):
        self._ckpt = ckpt

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self._image_index[i]

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'JPEGImages',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Layout',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join('data', 'VOCdevkit2007')

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_pascal_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
            #cPickle.dump(gt_roidb, fid)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print('{} ss roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        if self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print('wrote ss roidb to {}'.format(cache_file))

        return roidb

    def rpn_roidb(self):
        if self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print('loading {}'.format(filename))
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(cfg.DATA_DIR,
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
               'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
            keep = ds_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
            boxes = boxes[keep, :]
            box_list.append(boxes)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        weather = self._weather_to_ind[tree.find('weather').text.lower().strip()]
        altitude = self._altitude_to_ind[tree.find('altitude').text.lower().strip()]
        angle = self._angle_to_ind[tree.find('angle').text.lower().strip()]
        if not self.config['use_diff']:
            # Exclude the samples labeled as difficult
            non_diff_objs = [
                obj for obj in objs if int(obj.find('difficult').text) == 0]
            if len(non_diff_objs) != len(objs):
                print('Removed {} difficult objects'.format(
                    len(objs) - len(non_diff_objs)))
            objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'weather': weather,
                'altitude': altitude,
                'angle': angle,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
            else self._comp_id)
        return comp_id

    def _get_voc_results_file_template(self, path):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        return os.path.join(path, filename)

    def _write_voc_results_file(self, path, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Writing {} VOC results file'.format(cls))
            filename = self._get_voc_results_file_template(path).format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, 4],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _write_voc_results_file_attributes(self, path, all_boxes, attr_name):
        if attr_name == 'weather':
            attr_ind = 5
            attributes = self._weathers
        elif attr_name == 'altitude':
            attr_ind = 6
            attributes = self._altitudes
        else:
            attr_ind = 7
            attributes = self._angles

        for ind, attr in enumerate(attributes):
            print('Writing {} VOC results file'.format(attr))
            filename = self._get_voc_results_file_template(path).format(attr)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[1][im_ind]
                    #print('{} = {}'.format(int(np.sum(dets[:, attr_ind])), ind * dets.shape[0]))
                    if dets == [] or int(np.sum(dets[:, attr_ind])) != ind * dets.shape[0]:
                        continue
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, 4],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, path, output_dir = 'output', ovthresh=0.5):
        annopath = os.path.join(
            self._devkit_path,
            'UAV' + self._year,
            'Annotations',
            '{:s}.xml')
        imagesetfile = os.path.join(
            self._devkit_path,
            'UAV' + self._year,
            'ImageSets',
            'Layout',
            self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        # The PASCAL VOC metric changed in 2010
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        filename = '_det_' + self._image_set + '_ap.txt'
        with open(os.path.join(path, filename), 'wt') as f:
            eval_classes = [self._classes]
            if self._gamma_altitude > 1e-10:
                eval_classes.append(self._altitudes)
            if self._gamma_angle > 1e-10:
                eval_classes.append(self._angles)
            if self._gamma_weather > 1e-10:
                eval_classes.append(self._weathers)

            for classes in eval_classes:
                for i, cls in enumerate(classes):
                    if cls == '__background__':
                        continue
                    filename = self._get_voc_results_file_template(path).format(cls)
                    rec, prec, ap_use_07_metric, ap_no_use_07_metric = voc_eval(
                            filename, annopath, imagesetfile, cls, cachedir, ovthresh=ovthresh)
                    f.write('AP (07_metric) for {} = {:.4f}\n'.format(cls, ap_use_07_metric))
                    f.write('AP for {} = {:.4f}\n'.format(cls, ap_no_use_07_metric))
        # print('~~~~~~~~')
        # print('Results:')
        # for ap in aps:
        #     print('{:.3f}'.format(ap))
        # print('{:.3f}'.format(np.mean(aps)))
        # print('~~~~~~~~')
        # print('')
        # print('--------------------------------------------------------------')
        # print('Results computed with the **unofficial** Python eval code.')
        # print('Results should be very close to the official MATLAB eval code.')
        # print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        # print('-- Thanks, The Management')
        # print('--------------------------------------------------------------')

    def _do_python_eval_baseline(self, path, output_dir = 'output', ovthresh=0.5):
        annopath = os.path.join(
            self._devkit_path,
            'UAV' + self._year,
            'Annotations',
            '{:s}.xml')
        imagesetfile = os.path.join(
            self._devkit_path,
            'UAV' + self._year,
            'ImageSets',
            'Layout',
            self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        # The PASCAL VOC metric changed in 2010
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        filename = '_det_' + self._image_set + '_ap.txt'
        with open(os.path.join(path, filename), 'wt') as f:
            eval_classes = [self._classes, self._altitudes, self._angles, self._weathers]
            for classes in eval_classes:
                for i, cls in enumerate(classes):
                    if cls == '__background__':
                        continue
                    filename = self._get_voc_results_file_template(path).format(cls)
                    rec, prec, ap_use_07_metric, ap_no_use_07_metric = voc_eval(
                            filename, annopath, imagesetfile, cls, cachedir, ovthresh=ovthresh)
                    f.write('AP (07_metric) for {} = {:.4f}\n'.format(cls, ap_use_07_metric))
                    f.write('AP for {} = {:.4f}\n'.format(cls, ap_no_use_07_metric))

    def evaluate_detections(self, all_boxes, output_dir, nuisance_type, baseline_method=False, ovthresh=0.5):
        path = os.path.join('data/results', 'UAV' + self._year, nuisance_type, 'altitude={}_angle={}_weather={}'.format(str(self._gamma_altitude), str(self._gamma_angle), str(self._gamma_weather)), str(self._epoch), str(self._ckpt))
        if not os.path.exists(path):
            os.makedirs(path)
        self._write_voc_results_file(path, all_boxes)
        self._write_voc_results_file_attributes(path, all_boxes, attr_name='weather')
        self._write_voc_results_file_attributes(path, all_boxes, attr_name='altitude')
        self._write_voc_results_file_attributes(path, all_boxes, attr_name='angle')
        if not baseline_method:
            self._do_python_eval(path, output_dir, ovthresh)
        else:
            self._do_python_eval_baseline(path, output_dir, ovthresh)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True