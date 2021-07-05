"""------------------------------------

* File Name : detection.py

* Purpose :

* Creation Date : 30-06-2021

* Last Modified : 06:35:27 PM, 04-07-2021

* Created By : Trinh Quang Truong

------------------------------------"""

from utils.tools import convert_type_bbox

import numpy as np

def iou(bbox_a, bbox_b, type_a='voc', type_b='voc'):

    if type_a != 'voc':
        convert_type_bbox(bbox_a, type_a + '2voc')
    if type_b != 'voc':
        convert_type_bbox(bbox_a, type_b + '2voc')

    x_a = max(bbox_a[0], bbox_b[0])
    y_a = max(bbox_a[1], bbox_b[1])
    x_b = min(bbox_a[2], bbox_b[2])
    y_b = min(bbox_a[3], bbox_b[3])

    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)
    if inter_area <= 0:
        return -1

    bbox_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
    bbox_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)

    result = inter_area / float(bbox_a_area + bbox_b_area - inter_area)

    return result


def get_scores(bboxes_a, bboxes_b):

    scores = []
    for bbox_a in bboxes_a:
        # flag = True
        max_iou = -1
        for bbox_b in bboxes_b:
            iou_ab = iou(bbox_a, bbox_b)
            if iou_ab > max_iou:
                max_iou = iou_ab

        scores.append(max_iou)

    return scores


def precision_recall_curve(y_true, scores, thresholds):
    list_precisions = []
    list_recalls = []

    for threshold in thresholds:
        n_samples = len(scores)
        y_pred = np.zeros(n_samples)
        y_pred[scores >= threshold] = 1

        recall = [
            np.sum(y_pred[:idx]) / n_samples
            for idx in range(1, n_samples + 1)
        ]
        precision = [
            np.sum(y_pred[:idx]) / idx for idx in range(1, n_samples + 1)
        ]

        list_precisions.append(precision)
        list_recalls.append(recall)

    return list_precisions, list_recalls


def voc_ap(recall, precision):

    recall.insert(0, 0.0)  # insert 0.0 at begining of list
    recall.append(1.0)  # insert 1.0 at end of list
    max_recall = recall[:]
    precision.insert(0, 0.0)  # insert 0.0 at begining of list
    precision.append(0.0)  # insert 0.0 at end of list
    max_precision = precision[:]
    # matlab indexes start in 1 but python in 0, so I have to do:
    #     range(start=(len(max_precision) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #     range(start=(len(max_precision) - 2), end=-1, step=-1)
    for i in range(len(max_precision) - 2, -1, -1):
        max_precision[i] = max(max_precision[i], max_precision[i + 1])

    i_list = []
    for i in range(1, len(max_recall)):
        if max_recall[i] != max_recall[i - 1]:
            i_list.append(i)  # if it was matlab would be i + 1

    print(len(max_recall))
    ap = 0.0
    for i in i_list:
        ap += ((max_recall[i] - max_recall[i - 1]) * max_precision[i])
    return ap, max_recall, max_precision
    
