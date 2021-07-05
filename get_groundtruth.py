"""------------------------------------

* File Name : get_groundtruth.py

* Purpose :

* Creation Date : 13-06-2021

* Last Modified : 11:53:27 PM, 25-06-2021

* Created By : Trinh Quang Truong

------------------------------------"""

import os

import glob
import json
import cv2

from utils.tools import convert_type_bbox

#  from model.model_extraction import MobileFaceNet
#  from utils.classification import Fastmean

#  MODEL_EXTRACTION = MobileFaceNet()
#  MODEL_CLASSIFICATION = Fastmean()
#
#  img = cv2.cvtColor(cv2.imread('data/Tram/Tram2.jpg'), cv2.COLOR_BGR2RGB)
#  img2 = cv2.cvtColor(cv2.imread('data/Tram/Tram1.jpg'), cv2.COLOR_BGR2RGB)
#
#  vector_of_imgs = MODEL_EXTRACTION.transform([img, img2])
#  MODEL_CLASSIFICATION.fit(vector_of_imgs)
#
#  print(MODEL_CLASSIFICATION.labels)

#  list_path_images = [
#  j for i in glob.glob(os.path.join('data', '*'))
#  for j in glob.glob(os.path.join(i, '*.jpg'))
#  ]

list_path_images = [i for i in glob.glob(os.path.join('../data', '*.jpg'))]

list_of_annotations = []
list_of_images = []
for path in list_path_images:
    path_json = os.path.splitext(path)[0] + '.json'

    with open(path_json, 'r') as f:
        list_of_annotations += json.load(f)

    #  list_of_images.append(cv2.imread(path))

for idx, annotation in enumerate(list_of_annotations):
    img = cv2.imread(list_path_images[idx])

    list_bbox = [[
        int(j)
        for j in convert_type_bbox(list(i['coordinates'].values()), 'yolo2voc')
    ] for i in annotation['annotations']]

    list_label = [i['label'] for i in annotation['annotations']]
    print(list_label)
    for jdx, bbox in enumerate(list_bbox):
        img_crop = img[bbox[1]:bbox[3] + 1, bbox[0]:bbox[2] + 1]
        folder_path = os.path.join('img_groundtruth', list_label[jdx])
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)

        img_name = os.path.join(folder_path,
                                list_path_images[idx].split('/')[-1])

        cv2.imwrite(img_name, img_crop)
