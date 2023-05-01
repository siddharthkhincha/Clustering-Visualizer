import copy
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA

color_list = ['#001f3f', '#003300', '#8B0000', '#008b8b', '#8B008B', '#B8860B', '#2E2E2E', '#1C1C1C', '#8B2500', '#4B0082', '#A52A2A', '#8B5F65', '#4F4F4F', '#556B2F', '#800000', '#000080', '#008080', '#FF7F50', '#FFD700', '#4B0082', '#BDB76B', '#8B668B', '#32CD32', '#9932CC',
              '#8B4726', '#8B668B', '#FA8072', '#8B3A3A', '#8B6969', '#40E0D0', '#EE82EE', '#F5DEB3', '#7FFFD4', '#F0FFFF', '#F5F5DC', '#7FFF00', '#6495ED', '#DC143C', '#00008B', '#008B8B', '#B8860B', '#696969', '#006400', '#BDB76B', '#8B008B', '#556B2F', '#FF8C00', '#9932CC', '#8B0000', '#E9967A']


def kmeans(X, K, max_iters=100):
    # Initialize centroids randomly
    centroids = X[np.random.choice(X.shape[0], K, replace=False), :]
    labels = np.zeros(X.shape[0], dtype=int)
    all_centroids = []
    all_labels = []
    # Run iterations until convergence or maximum iterations are reached
    for i in range(max_iters):
        # Assign each data point to the closest centroid
        dists = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(dists, axis=0)

        # Update centroids based on the mean of the data points in each cluster
        for k in range(K):
            centroids[k] = X[labels == k].mean(axis=0)
        all_centroids.append(copy.deepcopy(centroids))
        all_labels.append(labels[:])
#         plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=100, c='r')
#         plt.show()
    return all_centroids, all_labels


def call_kmeans(dataset, num_clusters=3, max_iters=100):
    # os.chdir("..")
    os.system("rm -r Outputs/Kmeans")
    os.mkdir("Outputs/Kmeans")
    all_centroids, all_labels = kmeans(dataset, num_clusters, max_iters)
    for i in range(100):
        if i != 0 and np.array_equal(all_centroids[i], all_centroids[i-1]) and np.array_equal(all_labels[i], all_labels[i-1]):
            break
        centroids = all_centroids[i]
        labels = all_labels[i]

    #     plt.scatter(Y[:, 0], Y[:, 1], c=labels)
    #     plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='r',cmap=pylab.cm.gist_rainbow)
        colormap = [color_list[i] for i in labels]
        print(dataset[0])
        if len(dataset[0]) == 3:
            ax = plt.axes(projection="3d")
            ax.scatter(dataset[:, 0], dataset[:, 1], dataset[:, 2], c=colormap)
            ax.scatter(centroids[:, 0], centroids[:, 1],
                       centroids[:, 2], marker='*', s=200, c='r')
        else:
            plt.scatter(dataset[:, 0], dataset[:, 1], c=colormap)
            plt.scatter(centroids[:, 0], centroids[:, 1],
                        marker='*', s=200, c='r')
        plt.savefig('Outputs/Kmeans/output'+str(i)+".png")
        plt.clf()

    print("Kmeans done")
