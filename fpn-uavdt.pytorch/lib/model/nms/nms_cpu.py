from __future__ import absolute_import
import torch
import numpy as np

def rescore(overlap, scores, thresh, type='gaussian'):
    assert overlap.shape[0] == scores.shape[0]
    if type == 'linear':
        print("linear soft nms")
        inds = np.where(overlap >= thresh)[0]
        scores[inds] = scores[inds] * (1 - overlap[inds])
    else:
        print("gaussian soft nms being used...")
        scores = scores * np.exp(- overlap**2 / thresh)

    return scores


def soft_nms(dets, thresh, max_dets):
    #print("using soft nms...")
    dets = dets.cpu().numpy()
    if dets.shape[0] == 0:
        return np.zeros((0, 5))

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    scores = scores[order]

    if max_dets == -1:
        max_dets = order.size

    keep = np.zeros(max_dets, dtype=np.intp)
    keep_cnt = 0

    while order.size > 0 and keep_cnt < max_dets:
        i = order[0]
        dets[i, 4] = scores[0]
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        order = order[1:]
        scores = rescore(ovr, scores[1:], thresh, type = "linear")

        tmp = scores.argsort()[::-1]
        order = order[tmp]
        scores = scores[tmp]

        keep[keep_cnt] = i
        keep_cnt += 1

    keep = keep[:keep_cnt]
    dets = dets[keep, :]
    #print(keep.shape)
    ret = torch.from_numpy(keep).cuda().int()
    #print(type(ret))
    #return torch.IntTensor(dets.tolist())
    return ret



import numpy as np
import torch

def _nms_cpu(dets, thresh):
    dets = dets.cpu().numpy()
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order.item(0)
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    keep = torch.from_numpy(np.asarray(keep)).cuda().int()
    #return torch.IntTensor(keep)
    return keep


