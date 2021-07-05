"""------------------------------------

* File Name :

* Purpose :

* Creation Date : 02-06-2021

* Last Modified : 05:45:05 PM, 05-07-2021

* Created By : Trinh Quang Truong

------------------------------------"""

import numpy as np
import sys
from utils.tools import cal_distance


class Fastmean(object):
    """
            cluster same kmean online
    """
    def __init__(self,
                 n_clusters=0,
                 centroids=None,
                 data=None,
                 labels=[],
                 distance='cov',
                 threshold=0.38,
                 dimension=128,
                 update_centroids=True):
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
        self.update_centroids = update_centroids
        if centroids is None:
            self.centroids = np.zeros((n_clusters, 1, dimension))
        else:
            self.centroids = np.array(centroids)

        #  if data is None:
        #      self.data = np.zeros((n_clusters, 1, dimension))
        #  else:
        #  self.data = np.array(data)
        self.labels = np.array(labels, dtype=np.int32)

    def add_point(self, point, type_add='new_centroid', threshold=None):
        """
            add 1 point 
        Parameters:
        ---------
        point : array shape = (1, 128)
            point will be add
        threshold : float
            threshold in cluster
        ---------
        ---------
        Return
        ---------
        rank : array
            rank of centroids
        """
        if threshold is None:
            threshold = self.threshold
        if self.n_clusters == 0:
            self.centroids = np.append(self.centroids, [point], axis=0)
            self.labels = np.append(self.labels, self.n_clusters)
            self.n_clusters += 1
        else:
            #  min_distance = self.cal_distance(self.centroids[0], point)
            #  min_cluster = 0
            #  for cluster, centroid in enumerate(self.centroids[1:]):
            #      temp_distance = self.cal_distance(centroid, point,
            #                                        self.distance)
            #      if temp_distance < min_distance:
            #          min_distance = temp_distance
            #  min_cluster = cluster + 1
            distance_array = np.array([
                cal_distance(i, point, self.distance) for i in self.centroids
            ])
            sort_index = np.argsort(distance_array)
            #  print(distance_array)
            #  print(sort_index.shape, self.centroids.shape)
            rank = sort_index
            #  print(rank.shape)
            min_cluster = sort_index[0]
            min_distance = distance_array[min_cluster]
            #  print(min_distance)
            if min_distance <= threshold:
                self.labels = np.append(self.labels, min_cluster)
                #  self.data = np.array(self.data, [point], axis=0)
                #  self.centroids[min_cluster] = np.mean(
                #  self.data[self.labels == min_cluster], axis=0)
                num_of_member = np.sum(self.labels == min_cluster)
                if self.update_centroids:
                    self.centroids[min_cluster] = (
                        self.centroids[min_cluster] * num_of_member +
                        point) / (num_of_member + 1)
                return rank
            else:
                if type_add == 'new_centroid':
                    self.centroids = np.append(self.centroids, [point], axis=0)
                    self.labels = np.append(self.labels, self.n_clusters)
                    self.n_clusters += 1
                    return rank
                else:
                    return None


#              if min_distance <= threshold:
#      self.labels = np.append(self.labels, min_cluster)
#      #  self.data = np.array(self.data, [point], axis=0)
#      #  self.centroids[min_cluster] = np.mean(
#      #  self.data[self.labels == min_cluster], axis=0)
#      num_of_member = np.sum(self.labels == min_cluster)
#      self.centroids[min_cluster] = (
#          self.centroids[min_cluster] * num_of_member +
#          point) / (num_of_member + 1)
#      return True
#  else:
#      if type_add == 'new_centroid':
#          self.centroids = np.append(self.centroids, [point], axis=0)
#          self.labels = np.append(self.labels, self.n_clusters)
#          self.n_clusters += 1
#          return True
#      else:
#                      return False

    def fit(self, X, threshold=None):
        if threshold is None:
            threshold = self.threshold
        for point in X:
            point = np.array([point])
            if point.shape == (1, self.dimension):
                self.add_point(point, threshold=threshold)
            else:
                print(
                    'ERROR!!! shape vector is wrong! Require: (1, {})'.format(
                        self.dimension))
                sys.exit(1)
