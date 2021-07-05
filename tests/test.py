"""------------------------------------

* File Name : test.py

* Purpose :

* Creation Date : 01-06-2021

* Last Modified : 02:57:04 PM, 13-06-2021

* Created By : Trinh Quang Truong

------------------------------------"""

import cv2
import os
import glob
import numpy as np
from model.model_detection import MTCNNDetector, ULFDetector
from utils.alignment import alignment_img
from model.model_extraction import embedding_img
from utils.classification import tmean

#  import matplotlib.pyplot as plt
#  from mtcnn import MTCNN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#  mtcnn = MTCNNDetector()


def crop_and_label(img_folder,
                   img_type='jpg',
                   output_folder='img_crop',
                   threshold=0.38):
    #  abspath = os.path.dirname(os.path.abspath(__file__))
    ulf = ULFDetector()
    #  list_path_test = glob.glob(os.path.join(abspath, ))
    img_folder = os.path.join(img_folder, '*.' + img_type)
    list_path_test = glob.glob(img_folder)
    list_img_align = []
    for path in list_path_test:
        #  print(path)
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

        bboxesA, pointsA = ulf.predict(img)

        list_img_align += alignment_img(img, bboxesA, pointsA)

    list_embedding = embedding_img(list_img_align)

    classi = tmean(threshold=threshold)
    classi.fit(list_embedding)
    print(classi.labels)

    if os.path.isdir(output_folder):
        os.system('rm -fr $output_folder')
    os.mkdir(output_folder)
    arr_img = np.array(list_img_align)
    for i in range(classi.n_clusters):
        for j, img_crop in enumerate(arr_img[classi.labels == i]):
            output_path = os.path.join(output_folder,
                                       'label%d_%d.jpg' % (i, j))
            cv2.imwrite(output_path, cv2.cvtColor(img_crop, cv2.COLOR_RGB2BGR))
