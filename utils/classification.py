"""------------------------------------

* File Name :

* Purpose :

* Creation Date : 02-06-2021

* Last Modified : 05:50:55 PM, 03-06-2021

* Created By : Trinh Quang Truong

------------------------------------"""

import numpy as np
from numpy import dot
from numpy.linalg import norm
import sys

class tmean(object):
    """
            cluster same kmean online
    """
    def __init__(self,
                 n_clusters=0,
                 centroids=None,
                 data=None,
                 labels=[],
                 distance='cov',
                 threshold=0.7,
                 dimension=128):
        """
            initialize the tmean cluster
        Parameters:
        ----------
            n_clusters : int
                number of centroids
            centroids : list
                list of centroids
            labels : list
                labels of each point
            distance : {cov: Cosine, l1: Manhattan, l2: Euclidean}
            threshold : float
                threshold
            dimension : int
                dimension of vector embedding
        """
        self.n_clusters = n_clusters
        self.threshold = threshold
        self.distance = distance
        self.dimension = dimension
        if centroids is None:
            self.centroids = np.zeros((n_clusters, 1, dimension))
        else:
            self.centroids = np.array(centroids)

        #  if data is None:
        #      self.data = np.zeros((n_clusters, 1, dimension))
        #  else:
        #  self.data = np.array(data)
        self.labels = np.array(labels)

    def cal_distance(self, point1, point2, distance='cov'):
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

    def add_point(self, point, type_add='new_centroid', threshold=None):
        """
            add 1 point 
        Parameters:
        ---------
        point : array shape = (1, 128)
            point will be add
        threshold : float
            threshold in cluster
        """
        if threshold is None:
            threshold = self.threshold
        if self.n_clusters == 0:
            self.centroids = np.append(self.centroids, [point], axis=0)
            self.labels = np.append(self.labels, self.n_clusters)
            self.n_clusters += 1
        else:
            min_distance = self.cal_distance(self.centroids[0], point)
            min_cluster = 0
            for cluster, centroid in enumerate(self.centroids[1:]):
                temp_distance = self.cal_distance(centroid, point,
                                                  self.distance)
                if temp_distance < min_distance:
                    min_distance = temp_distance
                    min_cluster = cluster + 1
            if min_distance <= threshold:
                self.labels = np.append(self.labels, min_cluster)
                #  self.data = np.array(self.data, [point], axis=0)
                #  self.centroids[min_cluster] = np.mean(
                #  self.data[self.labels == min_cluster], axis=0)
                num_of_member = np.sum(self.labels == min_cluster)
                self.centroids[min_cluster] = (
                    self.centroids[min_cluster] * num_of_member +
                    point) / (num_of_member + 1)
                return True
            else:
                if type_add == 'new_centroid':
                    self.centroids = np.append(self.centroids, [point], axis=0)
                    self.labels = np.append(self.labels, self.n_clusters)
                    self.n_clusters += 1
                    return True
                else:
                    return False

    def fit(self, X, threshold=None):
        if threshold is None:
            threshold = self.threshold
        for point in X:
            point = np.array(point)
            if point.shape == (1, self.dimension):
                self.add_point(point, threshold=threshold)
            else:
                print(
                    'ERROR!!! shape vector is wrong! Require: (1, {})'.format(
                        self.dimension))
                sys.exit(1)
