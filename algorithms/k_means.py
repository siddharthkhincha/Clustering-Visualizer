import copy
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA

iris = datasets.load_iris()
# Y = iris["data"]

Y = PCA(n_components=3).fit_transform(iris.data)

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
    return all_centroids,all_labels

def call_kmeans(dataset, num_clusters=3, max_iters=100):
    # os.chdir("..")
    os.system("rm -r Outputs/Kmeans")
    os.mkdir("Outputs/Kmeans")
    all_centroids, all_labels = kmeans(dataset, num_clusters, max_iters)
    for i in range(100):
        if i!=0 and np.array_equal(all_centroids[i],all_centroids[i-1]) and np.array_equal(all_labels[i],all_labels[i-1]):
            break
        centroids = all_centroids[i]
        labels = all_labels[i]
        ax = plt.axes(projection ="3d")
    #     plt.scatter(Y[:, 0], Y[:, 1], c=labels)
    #     plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='r',cmap=pylab.cm.gist_rainbow)
        ax.scatter(Y[:, 0], Y[:, 1],Y[:,2], c=labels)
        ax.scatter(centroids[:, 0], centroids[:, 1],centroids[:, 2], marker='*', s=200, c='r')
        plt.savefig('Outputs/Kmeans/output'+str(i)+".png")
        plt.clf()
    
        
    print("Kmeans done")
        

if __name__ == "__main__":
    call_kmeans(Y)