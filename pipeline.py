"""------------------------------------

* File Name : face_recognition.py

* Purpose :

* Creation Date : 04-06-2021

* Last Modified : 02:57:40 PM, 05-07-2021

* Created By : Trinh Quang Truong

------------------------------------"""

import os

import glob
import cv2
import matplotlib.pyplot as plt

from model.model_detection import ULFDetector
from model.model_extraction import MobileFaceNet_Keras
from model.model_alignment import OnetLnet
from model.model_classification import Fastmean

DETECTOR = ULFDetector()
ALIGNMENT = OnetLnet()
EXTRACTOR = MobileFaceNet_Keras('model/weights/fine_tune.h5')
CLASSIFIER = Fastmean(threshold=0.3)

list_path_images = [
    j for i in glob.glob(os.path.join('data', '*'))
    for j in glob.glob(os.path.join(i, '*.jpg'))
]

list_of_images = [
    cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    for path in list_path_images
]

#  bboxes_and_points = [
#  list(results) for image in list_of_images
#  for results in DETECTOR.predict(image)
#  ]

# Face detection
list_of_bboxes = []
list_of_points = []
for image in list_of_images:
    bboxes, points = DETECTOR.predict(image)
    list_of_bboxes.append(bboxes)
    list_of_points.append(points)

# Face alignment
list_name_images = [
    os.path.splitext(os.path.basename(image))[0] for image in list_path_images
]
list_of_aligns = [
    (list_name_images[idx], j) for idx, img in enumerate(list_of_images)
    for j in ALIGNMENT.alignment_img(
        img, list_of_bboxes[idx], list_of_points[idx], crop_size=(96, 112))
]

# Face extraction
list_of_vectors = EXTRACTOR.transform([i[1] for i in list_of_aligns])

# Face classification
CLASSIFIER.fit(list_of_vectors)
labels = CLASSIFIER.labels

# Save face image
output_dir = 'img_crop_(112x96)'
if not os.path.isdir(output_dir):
    #  os.system('rm -fr $output_dir')
    os.mkdir(output_dir)

for idx, (name, image) in enumerate(list_of_aligns):
    output_path = os.path.join(output_dir,
                               'label{}_{}.jpg'.format(labels[idx], name))
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
