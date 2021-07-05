"""------------------------------------

* File Name : evaluation.py

* Purpose :

* Creation Date : 04-07-2021

* Last Modified : 11:14:15 PM, 05-07-2021

* Created By : Trinh Quang Truong

------------------------------------"""

import os
import sys

import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
from random import seed
from random import randint

from evaluate.evaluate_classification import evaluate_with_dir, evaluate_with_far
from model.model_detection import ULFDetector, MTCNNDetector
from model.model_extraction import MobileFaceNet_Keras, MobileFaceNet

#  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def far_dir(test_folder='img_groundtruth',
            model_folder='model/weights',
            min_threshold=0.1,
            max_threshold=1.4,
            step=0.1,
            show_plot=False):

    list_model_path = glob.glob(os.path.join(model_folder, '*.h5'))

    list_far = []
    list_dir = []

    thresholds = np.arange(min_threshold, max_threshold, step)

    for idx, path in enumerate(list_model_path):
        print('Evaluate ', path)
        EXTRACTION = MobileFaceNet_Keras(path)

        list_folder = glob.glob(os.path.join(test_folder, '*'))

        list_img = [[
            cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            for path in glob.glob(os.path.join(folder, '*.jpg'))
        ] for folder in list_folder]

        list_X = [EXTRACTION.transform(X) for X in list_img]

        far = evaluate_with_far(list_X, thresholds)
        dir = evaluate_with_dir(list_X, thresholds)

        list_dir.append(dir)
        list_far.append(far)

        if show_plot:
            plt.plot(thresholds, far, label='FAR', c='r')
            plt.plot(thresholds, dir, label='DIR', c='g')
            plt.xlabel('Thresholds')
            plt.ylabel('Scores')
            plt.legend()
            plt.savefig('far_dir_%d.png' % idx)

            plt.show()

    return thresholds, list_far, list_dir


def average_precision(type_model='ulf', path_img_folder='data'):

    if type_model == 'ulf':
        DETECTOR = ULFDetector()

    elif type_model == 'mtcnn':
        DETECTOR = MTCNNDetector()
    else:
        print('type_mode is wrong!')
        sys.exit(1)

    list_img = glob.glob(os.path.join(path_img_folder, '*'))
    list_bboxes = []
    list_points = []
    for img in list_img:
        bboxes, points = DETECTOR.predict(img)

        list_bboxes.append([i[:4] for i in bboxes])
        list_points.append(points)


def evaluate_mobilefacenet(type_mode='keras',
                           path_to_model='model/weights/MobileFacNet_9930.h5',
                           path_img_folder='img_groundtruth',
                           resume=False,
                           min_threshold=0.1,
                           max_threshold=1.4,
                           step=0.1,
                           iter=10):

    if type_mode == 'keras':

        EXTRACTOR = MobileFaceNet_Keras(path_to_model=path_to_model)
    else:
        EXTRACTOR = MobileFaceNet()

    list_folders = [
        i for i in glob.glob(os.path.join(path_img_folder, '*'))
        if os.path.isdir(i)
    ]

    list_paths = [glob.glob(os.path.join(i, '*.jpg')) for i in list_folders]

    if resume:
        with open('index.pkl', 'rb') as f:
            list_index = pickle.load(f)
    else:
        seed(1)
        list_index = [[randint(0,
                               len(i) - 1) for i in list_paths]
                      for _ in range(iter)]

        with open('index.pkl', 'wb') as f:
            pickle.dump(list_index, f)

    list_imgs = [[
        cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) for path in folder
    ] for folder in list_paths]

    list_vectors = [EXTRACTOR.transform(i) for i in list_imgs]

    # Add PCA here

    thresholds = np.arange(min_threshold, max_threshold, step)
    list_of_dir = [0 for i in thresholds]
    list_of_far = [0 for i in thresholds]
    for idx in range(iter):
        index = list_index[idx]
        list_vector_train = [
            list_vectors[i][index[i]] for i in range(len(index))
        ]
        list_vector_test = [
            np.array([
                vector for vector in j
                if not np.all(vector == list_vector_train[i])
            ]) for i, j in enumerate(list_vectors)
        ]
        #          y_true = [[idx for _ in range(len(i))]
        #            for idx, i in enumerate(list_img_path)]

        dir = evaluate_with_dir(list_vector_test,
                                list_vector_train,
                                thresholds=thresholds,
                                update_centroids=False)

        far = evaluate_with_far(list_vector_train,
                                thresholds=thresholds,
                                update_centroids=False)

        list_of_dir = [x + y for x, y in zip(dir, list_of_dir)]
        list_of_far = [x + y for x, y in zip(far, list_of_far)]

    list_of_far = [number / iter for number in list_of_far]
    list_of_dir = [number / iter for number in list_of_dir]

    return list_of_far, list_of_dir, thresholds


list_of_far, list_of_dir, thresholds = evaluate_mobilefacenet(iter=10)

print('far =', list_of_far)
print('dir =', list_of_dir)

# Show plot

plt.plot(thresholds, list_of_far, label='FAR', c='r')
plt.plot(thresholds, list_of_dir, label='DIR', c='g')
plt.xlabel('Thresholds')
plt.ylabel('Scores')
plt.legend()
plt.savefig('far_dir.png')

plt.show()

