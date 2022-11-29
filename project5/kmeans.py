'''kmeans.py
Performs K-Means clustering
Di Luo
CS 251 Data Analysis Visualization, Spring 2020
'''
import numpy as np
import matplotlib.pyplot as plt
from palettable import colorbrewer
from matplotlib.ticker import MaxNLocator


class KMeans():
    def __init__(self, data=None):
        '''KMeans constructor

        (Should not require any changes)

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features)
        '''

        # k: int. Number of clusters
        self.k = None
        # centroids: ndarray. shape=(k, self.num_features)
        #   k cluster centers
        self.centroids = None
        # data_centroid_labels: ndarray. shape=(self.num_samps,)
        #   Holds index of the assigned cluster of each data sample
        self.data_centroid_labels = None

        # inertia: float.
        #   Mean squared distance between each data sample and its assigned (nearest) centroid
        self.inertia = None

        # data: ndarray. shape=(num_samps, num_features)
        self.data = data
        # num_samps: int. Number of samples in the dataset
        self.num_samps = None
        # num_features: int. Number of features (variables) in the dataset
        self.num_features = None
        if data is not None:
            self.num_samps, self.num_features = data.shape

    def set_data(self, data):
        '''Replaces data instance variable with `data`.

        Reminder: Make sure to update the number of data samples and features!

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features)
        '''
        self.data = data
        self.num_samps = data.shape[0]
        self.num_features = data.shape[1]

    def get_data(self):
        '''Get a COPY of the data

        Returns:
        -----------
        ndarray. shape=(num_samps, num_features). COPY of the data
        '''
        return self.data.copy()

    def get_centroids(self):
        '''Get the K-means centroids

        (Should not require any changes)

        Returns:
        -----------
        ndarray. shape=(k, self.num_features).
        '''
        return self.centroids

    def get_data_centroid_labels(self):
        '''Get the data-to-cluster assignments

        (Should not require any changes)

        Returns:
        -----------
        ndarray. shape=(self.num_samps,)
        '''
        return self.data_centroid_labels

    def dist_pt_to_pt(self, pt_1, pt_2):
        '''Compute the Euclidean distance between data samples `pt_1` and `pt_2`

        Parameters:
        -----------
        pt_1: ndarray. shape=(num_features,)
        pt_2: ndarray. shape=(num_features,)

        Returns:
        -----------
        float. Euclidean distance between `pt_1` and `pt_2`.

        NOTE: Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running)
        '''
        return np.sqrt(np.sum(np.square(pt_1-pt_2)))
        

    def dist_pt_to_centroids(self, pt, centroids):
        '''Compute the Euclidean distance between data sample `pt` and and all the cluster centroids
        self.centroids

        Parameters:
        -----------
        pt: ndarray. shape=(num_features,)
        centroids: ndarray. shape=(C, num_features)
            C centroids, where C is an int.

        Returns:
        -----------
        ndarray. shape=(C,).
            distance between pt and each of the C centroids in `centroids`.

        NOTE: Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running)
        '''
        pt2 = pt[np.newaxis,:]
        return np.sqrt(np.sum(np.square(centroids-pt2), axis=1))

    def initialize(self, k):
        '''Initializes K-means by setting the initial centroids (means) to K unique randomly
        selected data samples

        Parameters:
        -----------
        k: int. Number of clusters

        Returns:
        -----------
        ndarray. shape=(k, self.num_features). Initial centroids for the k clusters.

        NOTE: Can be implemented without any for loops
        '''

        # Generate a list of k random indices without repetition from indices of rows of sel.data
        return np.take(self.data, np.random.randint(0, self.num_samps, k), axis=0)

    def initialize_plusplus(self, k):
        '''Initializes K-means by setting the initial centroids (means) according to the K-means++
        algorithm

        (LA section only)

        Parameters:
        -----------
        k: int. Number of clusters

        Returns:
        -----------
        ndarray. shape=(k, self.num_features). Initial centroids for the k clusters.

        TODO:
        - Set initial centroid (i = 0) to a random data sample.
        - To pick the i-th centroid (i > 0)
            - Compute the distance between all data samples and i-1 centroids already initialized.
            - Create the distance-based probability distribution (see notebook for equation).
            - Select the i-th centroid by randomly choosing a data sample according to the probability
            distribution.
        '''
        centroid_list = [] 
        pos = np.random.randint(0, self.num_samps, 1)
        centroid_list.append(pos)
        centroids = self.data[pos].reshape(1, self.num_features)
        for i in range(1, k): 
            dist = np.zeros([self.num_samps, i])
            for j in range(i):
                dist[:, j] = np.apply_along_axis(self.dist_pt_to_pt, 1, self.data, centroids[j])
            dist = np.min(dist, axis=1)
            dist = np.delete(dist, centroid_list)
            dist = dist * dist 
            dist = dist / np.sum(dist)
            pos = np.random.choice([i for i in range(self.num_samps) if i not in centroid_list], p=dist)
            centroid_list.append(pos)
            centroids = np.vstack([centroids, self.data[pos]])
        return np.array(centroids)

    def cluster(self, k=2, tol=1e-5, max_iter=1000, init_method='random', verbose=False):
        '''Performs K-means clustering on the data

        Parameters:
        -----------
        k: int. Number of clusters
        tol: float. Terminate K-means if the difference between all the centroid values from the
        previous and current time step < `tol`.
        max_iter: int. Make sure that K-means does not run more than `max_iter` iterations.
        verbose: boolean. Print out debug information if set to True.

        Returns:
        -----------
        self.inertia. float. Mean squared distance between each data sample and its cluster mean
        int. Number of iterations that K-means was run for

        TODO:
        - Initialize K-means variables
        - Do K-means as long as the max number of iterations is not met AND the difference between
        the previous and current centroid values is > `tol`
        - Set instance variables based on computed values.
        (All instance variables defined in constructor should be populated with meaningful values)
        - Print out total number of iterations K-means ran for
        '''
        self.k = k 
        n_iter = 0 

        if init_method == 'random':
            centroids = self.initialize(k)
        elif init_method == 'kmeans++':
            centroids = self.initialize_plusplus(k)

        while n_iter <= max_iter:
            labels = self.update_labels(centroids)
            centroids, diff = self.update_centroids(k, labels, centroids)
            n_iter += 1 
            if verbose:
                print(f'current iteration {n_iter}, max diff {np.max(diff)}')
            if np.max(diff) < tol:
                break
        if verbose: 
            print(f'total number of iterations: {n_iter}')
        self.inertia = self.compute_inertia()
        return self.inertia, n_iter

    def cluster_batch(self, k=2, n_iter=1, init_method='random', verbose=False):
        '''Run K-means multiple times, each time with different initial conditions.
        Keeps track of K-means instance that generates lowest inertia. Sets the following instance
        variables based on the best K-mean run:
        - self.centroids
        - self.data_centroid_labels
        - self.inertia

        Parameters:
        -----------
        k: int. Number of clusters
        tol: float. Terminate K-means if the difference between all the centroid values from the
        previous and current time step < `tol`.
        max_iter: int. Make sure that K-means does not run more than `max_iter` iterations.
        verbose: boolean. Print out debug information if set to True.
        '''
        optimized_inertia = float('inf')
        centroids = None 
        data_centroid_labels = None 
        count = 0
        for i in range(n_iter):
            inertia, n = self.cluster(k, init_method=init_method, verbose=verbose)
            count += n 
            if inertia < optimized_inertia:
                centroids = self.centroids
                data_centroid_labels = self.data_centroid_labels
                optimized_inertia = inertia
        self.centroids = centroids
        self.data_centroid_labels = data_centroid_labels
        self.inertia = optimized_inertia
        return count / n_iter

    def update_labels(self, centroids):
        '''Assigns each data sample to the nearest centroid

        Parameters:
        -----------
        centroids: ndarray. shape=(k, self.num_features). Current centroids for the k clusters.

        Returns:
        -----------
        ndarray. shape=(self.num_samps,). Holds index of the assigned cluster of each data sample

        Example: If we have 3 clusters and we compute distances to data sample i: [0.1, 0.5, 0.05]
        labels[i] is 2. The entire labels array may look something like this: [0, 2, 1, 1, 0, ...]
        '''

        self.centroids = centroids
        def idx(x):
            return np.argmin(self.dist_pt_to_centroids(x, centroids))
        self.data_centroid_labels = np.apply_along_axis(idx, 1, self.data)
        return self.data_centroid_labels

    def update_centroids(self, k, data_centroid_labels, prev_centroids):
        '''Computes each of the K centroids (means) based on the data assigned to each cluster

        Parameters:
        -----------
        k: int. Number of clusters
        data_centroid_labels. ndarray. shape=(self.num_samps,)
            Holds index of the assigned cluster of each data sample
        prev_centroids. ndarray. shape=(k, self.num_features)
            Holds centroids for each cluster computed on the PREVIOUS time step

        Returns:
        -----------
        new_centroids. ndarray. shape=(k, self.num_features).
            Centroids for each cluster computed on the CURRENT time step
        centroid_diff. ndarray. shape=(k, self.num_features).
            Difference between current and previous centroid values
        '''

        new_centroids = np.zeros([k, self.num_features])
        for i in range(k):
            data = self.data[data_centroid_labels == i]
            new_centroids[i] = np.average(data, axis=0)
        centroid_diff = new_centroids - prev_centroids
        self.centroids = new_centroids
        return new_centroids, centroid_diff

    def compute_inertia(self):
        '''Mean squared distance between every data sample and its assigned (nearest) centroid

        Parameters:
        -----------
        None

        Returns:
        -----------
        float. The average squared distance between every data sample and its assigned cluster centroid.
        '''
        SSE = 0
        for i in np.arange(self.num_samps):
            c_index = self.data_centroid_labels[i]
            SSE += np.square(self.dist_pt_to_pt(self.data[i], self.centroids[int(c_index)]))

        inertia = SSE/self.num_samps
        return inertia

    def plot_clusters(self):
        '''Creates a scatter plot of the data color-coded by cluster assignment.


        TODO:
        - Plot samples belonging to a cluster with the same color.
        - Plot the centroids in black with a different plot marker.
        - The default scatter plot color palette produces colors that may be difficult to discern
        (especially for those who are colorblind). Make sure you change your colors to be clearly
        differentiable.
            (LA Section): You should use a palette Colorbrewer2 palette. Pick one with a generous
            number of colors so that you don't run out if k is large (e.g. 10).
        '''
        brewer_colors = colorbrewer.qualitative.Set3_10.mpl_colormap
        fig, ax = plt.subplots()
        ax.scatter(self.data[:, 0], self.data[:, 1], c=self.data_centroid_labels, cmap=brewer_colors)
        ax.scatter(self.centroids[:, 0], self.centroids[:, 1], c='black', marker='+')

    def elbow_plot(self, max_k):
        '''Makes an elbow plot: cluster number (k) on x axis, inertia on y axis.

        Parameters:
        -----------
        max_k: int. Run k-means with k=1,2,...,max_k.

        TODO:
        - Run k-means with k=1,2,...,max_k, record the inertia.
        - Make the plot with appropriate x label, and y label, x tick marks.
        '''
        inertia = [] 
        for i in range(1, max_k):
            ita, num_iter = self.cluster(i, init_method='kmeans++')
            inertia.append(ita)
        x = np.linspace(1, max_k-1, max_k-1)
        fig, ax = plt.subplots()
        ax.plot(x, inertia)
        ax.set_xlabel('Num of clusters')
        ax.set_ylabel('Inertia')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    def replace_color_with_centroid(self):
        '''Replace each RGB pixel in self.data (flattened image) with the closest centroid value.
        Used with image compression after K-means is run on the image vector.

        Parameters:
        -----------
        None

        Returns:
        -----------
        None
        '''
        data = self.data.copy()
        centroids = self.centroids.astype(np.uint8)
        for i in range(self.k):
            data[self.data_centroid_labels==i] = centroids[i] 
        self.data = data
