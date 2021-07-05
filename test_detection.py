"""------------------------------------

* File Name : test_detection.py

* Purpose :

* Creation Date : 18-06-2021

* Last Modified : 06:43:06 PM, 04-07-2021

* Created By : Trinh Quang Truong

------------------------------------"""

import cv2
from model.model_detection import ULFDetector, MTCNNDetector

DETECTOR1 = ULFDetector()
DETECTOR2 = MTCNNDetector()

img = cv2.imread('data/BThoa/BThoa3.jpg')

result1, point1 = DETECTOR1.predict(img)
result2, point2 = DETECTOR2.predict(img)

print(result1, result2)
