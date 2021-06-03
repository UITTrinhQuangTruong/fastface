"""------------------------------------

* File Name :

* Purpose :

* Creation Date : 03-06-2021

* Last Modified : 05:36:38 PM, 03-06-2021

* Created By : Trinh Quang Truong

------------------------------------"""

import os
import cv2
import tensorflow as tf


def embedding_img(list_img_crop, model_path='weights/MobileFaceNet_9925_9680.pb'):
    abspath = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(abspath, model_path)
    model_exp = os.path.expanduser(model_dir)
    print(model_exp)

    with tf.compat.v1.Graph().as_default():
        with tf.compat.v1.Session() as sess:
            with tf.compat.v1.gfile.FastGFile(model_exp, 'rb') as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')

            inputs_placeholder = tf.compat.v1.get_default_graph(
            ).get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name(
                "embeddings:0")

            embedding_size = embeddings.get_shape()[1]
            print('Embedding size: {}'.format(embedding_size))

            # list_path_img = glob.glob(OUPUT_CROP_PATH_TEST + '*')
            emb_array = []
            for img_crop in list_img_crop:
                img = cv2.resize(img_crop, (112, 112))

                # Chuan hoa
                img = img - 127.5
                img = img * 0.0078125

                feed_dict = {inputs_placeholder: [img]}
                emb_array.append(sess.run(embeddings, feed_dict=feed_dict))

    print('Embedding {} images {} vectors'.format(len(list_img_crop),
                                                  len(emb_array)))

    return emb_array
