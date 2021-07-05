"""------------------------------------

* File Name : visualize.py

* Purpose :

* Creation Date : 14-06-2021

* Last Modified : 08:02:00 PM, 16-06-2021

* Created By : Trinh Quang Truong

------------------------------------"""

from model.model_extraction import MobileFaceNet
import tensorflow.compat.v1 as tf1

model = MobileFaceNet()

model.get_input_tensor()

model.get_layers_name()
