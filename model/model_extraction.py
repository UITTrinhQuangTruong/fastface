"""------------------------------------

* File Name :

* Purpose :

* Creation Date : 03-06-2021

* Last Modified : 02:48:04 PM, 05-07-2021

* Created By : Trinh Quang Truong

------------------------------------"""

import os

import cv2
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import numpy as np

from .helper import normalize_image


class MobileFaceNet(object):
    def __init__(self, model_path='weights/MobileFaceNet_9925_9680.pb'):
        tf1.reset_default_graph()
        #  tf1.disable_eager_execution()

        abspath = os.path.dirname(os.path.abspath(__file__))
        model_exp = os.path.join(abspath, model_path)

        self.load_model(model_exp)

    def load_model(self, model_path):

        self.graph = tf1.Graph()
        with tf1.gfile.FastGFile(model_path, 'rb') as f:
            self.graph_def = tf1.GraphDef()
            self.graph_def.ParseFromString(f.read())

        with self.graph.as_default():
            tf.import_graph_def(self.graph_def, name='')
            self.inputs = tf1.get_default_graph().get_tensor_by_name("input:0")

            self.embeddings = tf1.get_default_graph().get_tensor_by_name(
                "embeddings:0")

        self.sess = tf1.Session(graph=self.graph)

    def transform(self, X):
        """
            encoding face images into a 128-dimensional vector
        Parameters:
        ----------
            X : list or ndarray
                list of inputs (img)
        """
        list_of_imgs = [normalize_image(img) for img in X]
        return self.sess.run(self.embeddings, {self.inputs: list_of_imgs})

    def get_input_tensor(self):
        nodes = [
            n.name + ' => ' + n.op for n in self.graph_def.node
            if n.op in ('Placeholder')
        ]

        for node in nodes:
            print(node)

    def get_layers_name(self):
        layers = [op.name for op in self.graph.get_operations()]

        for layer in layers:
            print(layer)


class MobileFaceNet_Keras(object):
    def __init__(self, path_to_model='model/weights/MobileFacNet_9930.h5'):
        self.model = tf.keras.models.load_model(path_to_model)

    def transform(self, X):

        list_of_imgs = np.array([normalize_image(img, (96, 112)) for img in X])

        return self.model.predict(list_of_imgs)
