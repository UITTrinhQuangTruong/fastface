"""------------------------------------

* File Name : test.py

* Purpose :

* Creation Date : 01-06-2021

* Last Modified : 06:08:53 PM, 03-06-2021

* Created By : Trinh Quang Truong

------------------------------------"""

import cv2
import os
#  import matplotlib.pyplot as plt
#  from mtcnn import MTCNN
import glob
from model.model_detection import MTCNNDetector, ULFDetector
from utils.alignment import alignment_img
from model.model_extraction import embedding_img
from utils.classification import tmean

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
mtcnn = MTCNNDetector()
ulf = ULFDetector()

abspath = os.path.dirname(os.path.abspath(__file__))

list_path_test = sorted(glob.glob(os.path.join(abspath, '../data/*.jpg')))
list_img_align = []
for path in list_path_test[:10]:
    print(path)
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    bboxesA, pointsA = ulf.predict(img)

    #  model = MTCNN()
    #  bboxesA = model.detect_faces(img)
    #  img = cv2.cvtColor(cv2.imread(os.path.join(abspath, 'test.jpg')),
    #  cv2.COLOR_BGR2RGB)

    list_img_align += alignment_img(img, bboxesA, pointsA)

list_embedding = embedding_img(list_img_align)

classi = tmean(threshold=0.4)
classi.fit(list_embedding)
print(classi.labels)
