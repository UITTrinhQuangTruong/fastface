import os
import sys

import glob
from random import seed, choice
from numpy import dot
from numpy.linalg import norm
import numpy as np

def convert_type_bbox(bbox, type_bbox='yolo2voc'):
    if type_bbox == 'yolo2voc':
        '''
        [x_center, y_center, w, h] -> [x_min, y_min, x_max, y_max]
        '''
        w_2 = float(bbox[2] / 2)
        h_2 = float(bbox[3] / 2)
        return [bbox[0] - w_2, bbox[1] - h_2, bbox[0] + w_2, bbox[1] + h_2]

    elif type_bbox == 'voc2yolo':
        '''
        [x_min, y_min, x_max, y_max] -> [x_center, y_center, w, h]

        '''
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return [float(bbox[0] + w / 2), float(bbox[1] + h / 2), w, h]
    elif type_bbox == 'coco2voc':
        '''
        [x_min, y_min, w, h] -> [x_min, y_min, x_max, y_max]
        '''
        return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]


def cal_distance(point1, point2, distance='cov'):
    """
            calculate distance between point1 and point2
        Parameters:
        ---------
        point1, point2 : numpy array
        distance : {cov: Cosine, l1: Manhattan, l2: Euclidean}
"""
    point1_, point2_ = point1[0].T, point2[0].T
    if distance == 'cov':
        return 1 - dot(point1_, point2_) / (norm(point1_) * norm(point2_))
    elif distance == 'l1':
        return np.sum(np.abs(point1_ - point2_))
    elif distance == 'l2':
        return norm(point1_ - point2_)
    else:
        print("ERROR!!! distance in {'cov', 'l1', 'l2'}")
        sys.exit(1)


def create_test_set(folder='.',
                    type_data='Dir',
                    type_file='jpg',
                    numbers_of_test=10,
                    save=True):
    seed(1)
    if type_data == 'Dir':
        list_of_folders = [
            i for i in glob.glob(os.path.join(folder, '*')) if os.path.isdir(i)
        ]

        list_of_files = [
            glob.glob(os.path.join(i, '*.{}'.format(type_file)))
            for i in list_of_folders
        ]

        list_of_trainset = []
        list_of_testset = []
        for i in range(numbers_of_test):
            list_of_trainset.append([choice(i) for i in list_of_files])
            #  print(list_of_trainset)
            list_of_testset.append(
                [[path for path in folder if i != list_of_trainset[i][idx]]
                 for idx, folder in enumerate(list_of_files)])

        return list_of_files, list_of_trainset, list_of_testset
