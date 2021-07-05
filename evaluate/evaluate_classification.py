"""------------------------------------

* File Name : evaluate_classification.py

* Purpose :

* Creation Date : 16-06-2021

* Last Modified : 10:27:32 PM, 05-07-2021

* Created By : Trinh Quang Truong

------------------------------------"""

import numpy as np

from model.model_classification import Fastmean


def false_alarm_rate(list_X, iter=10, threshold=0.38, update_centroids=True):
    far = 0

    for i in range(iter):

        model = Fastmean(threshold=threshold,
                         update_centroids=update_centroids)
        model.fit(list_X)
        num_missing = len(list_X) - model.n_clusters
        far += num_missing / len(list_X)

    return far / iter


def evaluate_with_far(list_X, thresholds, iter=10, update_centroids=True):

    results = []
    for threshold in thresholds:
        results.append(
            false_alarm_rate(list_X,
                             iter=iter,
                             threshold=threshold,
                             update_centroids=update_centroids))

    return results


def detection_indentification_rate(list_X,
                                   list_centroids,
                                   iter=10,
                                   threshold=0.38,
                                   r=1,
                                   update_centroids=True):
    list_X_test = [i for i in list_X if i.shape[0] > 0]
    #  list_X_test = list_X
    #  print(list_X_test[0])

    result = 0
    for i in range(iter):
        #  X = get_test_unknown(list_X_test)
        #  #  print(np.all(np.isin(list_X_test[0][0], X)))
        #
        #  list_not_X = [[a for a in j if not np.all(a == X)]
        #                        for j in list_X_test]
        model = Fastmean(threshold=threshold,
                         update_centroids=update_centroids)
        model.fit(list_centroids)
        labels = model.labels
        #          if model.n_clusters == len(X):
        #      print(threshold, 'True')
        #  else:
        #              print(threshold, 'False')
        dir = 0
        s = 0
        for label, points in zip(labels, list_X_test):
            for point in points:
                #  print(point.shape)
                s += 1
                rank = model.add_point(np.array([point]), threshold=threshold)
                for x in rank[:r]:
                    if x == label:
                        dir += 1
                        break
        result += float((s - dir) / s)

    return result / float(iter)


def evaluate_with_dir(list_X_test,
                      list_centroids,
                      thresholds,
                      iter=10,
                      r=1,
                      update_centroids=True):
    results = []

    for threshold in thresholds:
        results.append(
            detection_indentification_rate(list_X_test,
                                           list_centroids,
                                           iter=iter,
                                           threshold=threshold,
                                           r=r,
                                           update_centroids=update_centroids))

    return results
