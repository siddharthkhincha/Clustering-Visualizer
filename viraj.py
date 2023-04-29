#!/usr/bin/env python
# coding: utf-8

# In[47]:


import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
import copy
from sklearn.cluster import AffinityPropagation
import pylab
import os
from tkinter import *
from tkinter import filedialog
import tkinter as tk
import re


# In[38]:


sns.set()


# In[39]:


iris = datasets.load_iris()
# Y = iris["data"]


# In[40]:


Y = PCA(n_components=2).fit_transform(iris.data)


# In[48]:


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
    
#     plt.show()


# In[42]:


import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

class AffinityPropagation():
    #!/usr/bin/env python
    # coding: utf-8
    #idk if assuming x as 2D maybe or not

    global A,R,S
    

    #To create dataset
    # np.random.seed(3)
    # n = 20
    # size = (n, 2)

    # x = np.random.normal(0, 1, size)

    # for i in range(4):
    #     center = np.random.rand(2) * 10
    #     x = np.append(x, np.random.normal(center, .5, size), axis=0)

    #     C = [c for s in [v * n for v in 'bgrcmyk'] for c in list(s)]
    #     temp = x.shape[0]
    #     print(temp)
    #     plt.figure(figsize=(15, 6))
    #     plt.title('Some clusters in 2d space')
    #     plt.scatter(x[:, 0], x[:, 1], c = C[:temp])
    #     plt.show()

    
    # In[21]:


    def similarity(self,xi, xj):
        return -((xi - xj)**2).sum()

    def create_matrices(self):
        S = np.zeros((x.shape[0], x.shape[0]))
        R = np.array(S)
        A = np.array(S)
        print(S.shape[0])
        # compute similarity for every data point.
        for i in range(x.shape[0]):
            for k in range(x.shape[0]):
                S[i, k] = self.similarity(x[i], x[k])
                
        return A, R, S


    # In[22]:


    # def update_r(damping=0.9):
    #     global R
    #     for i in range(x.shape[0]):
    #         for k in range(x.shape[0]):
    #             v = S[i, :] + A[i, :]
    #             v[k] = -np.inf
    #             v[i]= -np.inf
    #             R[i, k] = R[i, k] * damping + (1 - damping) * (S[i, k] - np.max(v))


    # In[23]:


    # A, R, S = create_matrices()
    # %timeit update_r()


    # In[35]:
    global R
    # R = create_matrices()
    def update_r(self,damping=0.9):
        global R
        v = S + A
        rows = np.arange(x.shape[0])
        # We only compare the current point to all other points, 
        # so the diagonal can be filled with -infinity
        np.fill_diagonal(v, -np.inf)

        # max values
        idx_max = np.argmax(v, axis=1)
        first_max = v[rows, idx_max]

        # Second max values. For every column where k is the max value.
        v[rows, idx_max] = -np.inf
        second_max = v[rows, np.argmax(v, axis=1)]

        # Broadcast the maximum value per row over all the columns per row.
        max_matrix = np.zeros_like(R) + first_max[:, None]
        max_matrix[rows, idx_max] = second_max

        new_val = S - max_matrix

        R = R * damping + (1 - damping) * new_val


    # In[36]:


    # A, R, S = create_matrices()
    # get_ipython().run_line_magic('timeit', 'update_r()')


    # In[34]:





    # In[39]:

    global A
    def update_a(self,damping = 0.9):
        global A
        k_k_idx = np.arange(x.shape[0])
        # set a(i, k)
        a = np.array(R)
        a[a < 0] = 0
        np.fill_diagonal(a, 0)
        a = a.sum(axis=0) # columnwise sum
        a = a + R[k_k_idx, k_k_idx]

        # broadcasting of columns 'r(k, k) + sum(max(0, r(i', k))) to rows.
        a = np.ones(A.shape) * a

        # For every column k, subtract the positive value of k. 
        # This value is included in the sum and shouldn't be
        a -= np.clip(R, 0, np.inf)
        a[a > 0] = 0

        # set(a(k, k))
        w = np.array(R)
        np.fill_diagonal(w, 0)

        w[w < 0] = 0

        a[k_k_idx, k_k_idx] = w.sum(axis=0) # column wise sum
        A = A * damping + (1 - damping) * a


    # In[40]:


    # A,R,S = create_matrices()
    # get_ipython().run_line_magic('timeit', 'update_a()')


    # In[49]:


    def plot_iteration(self,A, R,iteration):
        fig = plt.figure(figsize=(12, 6))
        sol = A + R
        # every data point i chooses the maximum index k
        labels = np.argmax(sol, axis=1)
        exemplars = np.unique(labels)
        colors = dict(zip(exemplars, cycle('bgrcmyk')))
        
        for i in range(len(labels)):
            X = x[i][0]
            Y = x[i][1]
            
            if i in exemplars:
                exemplar = i
                edge = 'k'
                ms = 10
            else:
                exemplar = labels[i]
                ms = 3
                edge = None
                plt.plot([X, x[exemplar][0]], [Y, x[exemplar][1]], c=colors[exemplar])
            plt.plot(X, Y, 'o', markersize=ms,  markeredgecolor=edge, c=colors[exemplar])
            

        plt.title('Number of exemplars:' + str(len(exemplars)) + ' in iteration' + str(iteration) )
        plt.savefig('Outputs/AffinityPropogations/output'+str(iteration)+".png")
        plt.clf()
        return fig, labels, exemplars


    # In[50]:

    def __init__(self,X,damp= 0.7,max_iters = 100):
        global x 
        x = X
        damping = damp
        iterations = max_iters
        global A
        global R
        global S
        A, R, S = self.create_matrices()
        preference = np.median(S)
        np.fill_diagonal(S, preference)
        # damping = 0.5

        figures = []
        last_sol = np.ones(A.shape)
        last_exemplars = np.array([])

        c = 0
        last_i = 0
        for i in range(iterations):
            self.update_r(damping)
            self.update_a(damping)
            
            sol = A + R
            exemplars = np.unique(np.argmax(sol, axis=1))
            
            if last_exemplars.size != exemplars.size or np.all(last_exemplars != exemplars):
                fig, labels, exemplars = self.plot_iteration(A, R, i)
                figures.append(fig)
                last_i = i
            else :
                print("Same image of " + str(i) +  " as output" + str(last_i))



            if np.allclose(last_sol, sol):
                print(exemplars, i)
                break
                
            last_sol = sol
            last_exemplars = exemplars


# In[ ]:







    

# In[43]:


import numpy

def dbscan(D, eps=0.5, MinPts=5):
    '''
    Cluster the dataset `D` using the DBSCAN algorithm.
    
    dbscan takes a dataset `D` (a list of vectors), a threshold distance
    `eps`, and a required number of points `MinPts`.
    
    It will return a list of cluster labels. The label -1 means noise, and then
    the clusters are numbered starting from 1.
    '''
 
    # This list will hold the final cluster assignment for each point in D.
    # There are two reserved values:
    #    -1 - Indicates a noise point
    #     0 - Means the point hasn't been considered yet.
    # Initially all labels are 0.    
    labels = [0]*len(D)
    all_labels = []
    # C is the ID of the current cluster.    
    C = 0
    
    # This outer loop is just responsible for picking new seed points--a point
    # from which to grow a new cluster.
    # Once a valid seed point is found, a new cluster is created, and the 
    # cluster growth is all handled by the 'expandCluster' routine.
    
    # For each point P in the Dataset D...
    # ('P' is the index of the datapoint, rather than the datapoint itself.)
    for P in range(0, len(D)):
        # Only points that have not already been claimed can be picked as new 
        # seed points.    
        # If the point's label is not 0, continue to the next point.
        if not (labels[P] == 0):
           continue
        all_labels.append(copy.deepcopy(labels))
        # Find all of P's neighboring points.
        NeighborPts = region_query(D, P, eps)
        
        # If the number is below MinPts, this point is noise. 
        # This is the only condition under which a point is labeled 
        # NOISE--when it's not a valid seed point. A NOISE point may later 
        # be picked up by another cluster as a boundary point (this is the only
        # condition under which a cluster label can change--from NOISE to 
        # something else).
        if len(NeighborPts) < MinPts:
            labels[P] = -1
        # Otherwise, if there are at least MinPts nearby, use this point as the 
        # seed for a new cluster.    
        else: 
           C += 1
           grow_cluster(D, labels, P, NeighborPts, C, eps, MinPts)
    
    # All data has been clustered!
    return all_labels


def grow_cluster(D, labels, P, NeighborPts, C, eps, MinPts):
    '''
    Grow a new cluster with label `C` from the seed point `P`.
    
    This function searches through the dataset to find all points that belong
    to this new cluster. When this function returns, cluster `C` is complete.
    
    Parameters:
      `D`      - The dataset (a list of vectors)
      `labels` - List storing the cluster labels for all dataset points
      `P`      - Index of the seed point for this new cluster
      `NeighborPts` - All of the neighbors of `P`
      `C`      - The label for this new cluster.  
      `eps`    - Threshold distance
      `MinPts` - Minimum required number of neighbors
    '''

    # Assign the cluster label to the seed point.
    labels[P] = C
    
    # Look at each neighbor of P (neighbors are referred to as Pn). 
    # NeighborPts will be used as a FIFO queue of points to search--that is, it
    # will grow as we discover new branch points for the cluster. The FIFO
    # behavior is accomplished by using a while-loop rather than a for-loop.
    # In NeighborPts, the points are represented by their index in the original
    # dataset.
    i = 0
    while i < len(NeighborPts):    
        
        # Get the next point from the queue.        
        Pn = NeighborPts[i]
       
        # If Pn was labelled NOISE during the seed search, then we
        # know it's not a branch point (it doesn't have enough neighbors), so
        # make it a leaf point of cluster C and move on.
        if labels[Pn] == -1:
           labels[Pn] = C
        
        # Otherwise, if Pn isn't already claimed, claim it as part of C.
        elif labels[Pn] == 0:
            # Add Pn to cluster C (Assign cluster label C).
            labels[Pn] = C
            
            # Find all the neighbors of Pn
            PnNeighborPts = region_query(D, Pn, eps)
            
            # If Pn has at least MinPts neighbors, it's a branch point!
            # Add all of its neighbors to the FIFO queue to be searched. 
            if len(PnNeighborPts) >= MinPts:
                NeighborPts = NeighborPts + PnNeighborPts
            # If Pn *doesn't* have enough neighbors, then it's a leaf point.
            # Don't queue up it's neighbors as expansion points.
            #else:
                # Do nothing                
                #NeighborPts = NeighborPts               
        
        # Advance to the next point in the FIFO queue.
        i += 1        
    
    # We've finished growing cluster C!


def region_query(D, P, eps):
    '''
    Find all points in dataset `D` within distance `eps` of point `P`.
    
    This function calculates the distance between a point P and every other 
    point in the dataset, and then returns only those points which are within a
    threshold distance `eps`.
    '''
    neighbors = []
    
    # For each point in the dataset...
    for Pn in range(0, len(D)):
        
        # If the distance is below the threshold, add it to the neighbors list.
        if numpy.linalg.norm(D[P] - D[Pn]) < eps:
           neighbors.append(Pn)
            
    return neighbors


# In[49]:
def call_kmeans(Y):
    os.system("rm -r Outputs/Kmeans")
    os.mkdir("Outputs/Kmeans")
    all_centroids, all_labels = kmeans(Y, K=3)
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


def call_dbscan(Y):
    os.system("rm -r Outputs/DBScan")
    os.mkdir("Outputs/DBScan")

    all_labels = dbscan(Y)
    for i in range(len(all_labels)):
        labels = all_labels[i]
        ax = plt.axes(projection ="3d")
    #     plt.scatter(Y[:, 0], Y[:, 1], c=labels,cmap=pylab.cm.gist_rainbow)
        ax.scatter(Y[:, 0], Y[:, 1],Y[:,2], c=labels)
        plt.savefig('Outputs/DBScan/output'+str(i)+".png")
        plt.clf()

def call_affinity():
    os.system("rm -r Outputs/AffinityPropogations")
    os.mkdir("Outputs/AffinityPropogations")

    AffinityPropagation(Y,damp=0.9)





if __name__ == "__main__":
    call_affinity()