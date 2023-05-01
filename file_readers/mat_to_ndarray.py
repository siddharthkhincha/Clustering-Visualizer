import csv
import numpy as np
from sklearn.decomposition import PCA
import scipy.io

def read_mat(filename, header=True, reduce_dimensionality=True, n_components=3):

    
    # convert the data into a numpy array
    data = scipy.io.loadmat(filename)
    data = data['D']
    if reduce_dimensionality:
        data = PCA(n_components=n_components).fit_transform(data)

    return data

if __name__ == "__main__":
    # print the type of the input
    Y = read_mat('/home/techniche/Desktop/PLL_Clustering-Reformed-File-Structure/data_TwoDiamonds.mat', header=True, reduce_dimensionality=False)
    # debug
    # print type of Y
    print(type(Y))
    # print shape of Y
    print(Y.shape)
    # print first 5 rows of Y
    print(Y[:5])

