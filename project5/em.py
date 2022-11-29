'''em.py
Cluster data using the Expectation-Maximization (EM) algorithm with Gaussians
Di Luo
CS 251 Data Analysis Visualization, Spring 2020

Acknowledgement: Received help from Qingbo Liu
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib.colors import LogNorm
from scipy.special import logsumexp
from IPython.display import display, clear_output


class EM():
    def __init__(self, data=None):
        '''EM object constructor.
        See docstrings of individual methods for what these variables mean / their shapes

        (Should not require any changes)
        '''
        self.k = None
        self.centroids = None
        self.cov_mats = None
        self.responsibilities = None
        self.data_centroid_labels = None

        self.loglikelihood_hist = None

        self.data = data
        self.num_samps = None
        self.num_features = None
        if data is not None:
            self.num_samps, self.num_features = data.shape

    def gaussian(self, pts, mean, sigma):
        '''(LA section)
        Evaluates a multivariate Gaussian distribution described by
        mean `mean` and covariance matrix `sigma` at the (x, y) points `pts`

        Parameters:
        -----------
        pts: ndarray. shape=(num_samps, num_features).
            Data samples at which we want to evaluate the Gaussian
            Example for 2D: shape=(num_samps, 2)
        mean: ndarray. shape=(num_features,)
            Mean of Gaussian (i.e. mean of one cluster). Same dimensionality as data
            Example for 2D: shape=(2,) for (x, y)
        sigma: ndarray. shape=(num_features, num_features)
            Covariance matrix of a Gaussian (i.e. covariance of one cluster).
            Example for 2D: shape=(2,2). For standard deviations (sigma_x, sigma_y) and constant c,
                Covariance matrix: [[sigma_x**2, c*sigma_x*sigma_y],
                                    [c*sigma_x*sigma_y, sigma_y**2]]

        Returns:
        -----------
        ndarray. shape=(num_samps,)
            Multivariate gaussian evaluated at the data samples `pts`
        '''
        N,M = pts.shape
        det = np.sqrt(np.linalg.det(sigma))
        coef = np.power(np.sqrt(2*np.pi), M)
        exp = lambda x: np.exp(-0.5 * (x - mean) @ np.linalg.inv(sigma) @ (x - mean).T)
        f = lambda x : exp(x) / (coef * det)
        return np.apply_along_axis(f, 1, pts).reshape(N,)

    def initialize(self, k, method = 'random'):
        '''Initialize all variables used in the EM algorithm.

        Parameters:
        -----------
        k: int. Number of clusters.
        method: string. The method of initialization. Can be random or k++.

        Returns
        -----------
        None

        TODO:
        - Set k as an instance variable.
        - Initialize the log likelihood history to an empty Python list.
        - Initialize the centroids to random data samples
            shape=(k, num_features)
        - Initialize the covariance matrices to the identity matrix
        (1s along main diagonal, 0s elsewhere)
            shape=(k, num_features, num_features)
        - Initialize the responsibilities to an ndarray of 1s.
            shape=(k, num_samps)
        - Initialize the pi array (proportion of points assigned to each cluster) so that each cluster
        is equally likely.
            shape=(k,)
        '''
        self.k = k 
        self.loglikelihood_hist = [] 
        if method == 'random': 
            self.centroids = np.take(self.data, np.random.randint(0, self.num_samps, k), axis=0)
        elif method == 'kmeans++': 
            self.centroids = self.initialize_plusplus(k)
        self.cov_mats = np.array([np.identity(self.num_features) for i in range(k)])
        self.responsibilities = np.ones([k, self.num_samps])
        self.pi = np.ones(k)/np.sum(np.ones(k))

    # For extension 2: k++
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
        
    # For extension 2: k++
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

    # For extension 2: k++
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

    def e_step(self):
        '''Expectation (E) step in the EM algorithm.
        Set self.responsibilities, the probability that each data point belongs to each of the k clusters.
        i.e. leverages the Gaussian distribution.

        NOTE: Make sure that you normalize so that the probability that each data sample belongs
        to any cluster equals 1.

        Parameters:
        -----------
        None

        Returns
        -----------
        self.responsibilities: ndarray. shape=(k, num_samps)
            The probability that each data point belongs to each of the k clusters.
        '''
        for i in range(self.k):
            self.responsibilities[i,:] = self.pi[i]*self.gaussian(self.data, self.centroids[i], self.cov_mats[i])
        self.responsibilities = self.responsibilities / np.sum(self.responsibilities, axis=0)
        return self.responsibilities

    def m_step(self):
        '''Maximization (M) step in the EM algorithm.
        Set self.centroids, self.cov_mats, and self.pi, the parameters that define each Gaussian
        cluster center and spread, as well as the degree to which data points "belong" to each cluster

        TODO:
        - Compute the proportion of data points that belong to each cluster.
        - Compute the mean of each cluster. This is the mean over all data points, but weighting
        the data by the probability that they belong to that cluster.
        - Compute the covariance matrix of each cluster. Use the usual equation (for all the data),
        but before summing across data samples, make sure to weight each data samples by the
        probability that they belong to that cluster.

        NOTE: When computing the covariance matrix, use the updated cluster centroids for
        the CURRENT time step.

        Parameters:
        -----------
        None

        Returns
        -----------
        self.centroids: ndarray. shape=(k, num_features)
            Mean of each of the k Gaussian clusters
        self.cov_mats: ndarray. shape=(k, num_features, num_features)
            Covariance matrix of each of the k Gaussian clusters
            Example of a covariance matrix for a single cluster (2D data): [[1, 0.2], [0.2, 1]]
        self.pi: ndarray. shape=(k,)
            Proportion of data points belonging to each cluster.
        '''
        self.pi = np.sum(self.responsibilities, axis=1) / self.num_samps

        def helper1(x):
            a = x[-1]
            b = x[:-1].reshape(1, self.num_features)
            return a*(b-mean).T @ (b-mean)

        for i in range(self.k):
            R = np.sum(self.responsibilities[i])
            w = self.responsibilities[i].reshape(1, self.num_samps)
            self.centroids[i] = np.sum(w @ self.data, axis=0) / R
            mean = self.centroids[i]
            cov = np.apply_along_axis(helper1, 1, np.column_stack([self.data, w.T]))
            self.cov_mats[i] = np.sum(cov, axis=0) / R

        return self.centroids, self.cov_mats, self.pi

    def log_likelihood(self):
        '''Compute the sum of the log of the Gaussian probability of each data sample in each cluster
        Used to determine whether the EM algorithm is converging.

        Parameters:
        -----------
        None

        Returns
        -----------
        float. Summed log-likelihood of all data samples

        NOTE: Remember to weight each cluster's Gaussian probabilities by the proportion of data
        samples that belong to each cluster (pi).
        '''
        def helper2(x):
            s = np.apply_along_axis(lambda i: self.pi[i] * self.gaussian(x.reshape(1, self.num_features), self.centroids[i], self.cov_mats[i]), 1, np.arange(self.k).reshape(self.k, 1))
            return np.log(np.sum(s))
        
        return np.sum(np.apply_along_axis(helper2, 1, self.data))

    def cluster(self, k, max_iter=100, stop_tol=1e-3, verbose=False, animate=False, method = 'random'):
        '''Main method used to cluster data using the EM algorithm
        Perform E and M steps until the change in the loglikelihood from last step to the current
        step <= `stop_tol` OR we reach the maximum number of allowed iterations (`max_iter`).

        Parameters:
        -----------
        k: int. Number of clusters.
        max_iter: int. Max number of iterations to allow the EM algorithm to run.
        stop_tol: float. Stop running the EM algorithm if the change of the loglikelihood from the
        previous to current step <= `stop_tol`.
        verbose: boolean. If true, print out the current iteration, current log likelihood,
            and any other helpful information useful for debugging.

        Returns:
        -----------
        self.loglikelihood_hist: Python list. The log likelihood at each iteration of the EM algorithm.

        NOTE: Reminder to initialize all the variables before running the EM algorithm main loop.
            (Use the method that you wrote to do this)
        NOTE: At the end, print out the total number of iterations that the EM algorithm was run for.
        NOTE: The log likelihood is a NEGATIVE float, and should increase (approach 0) if things are
            working well.
        '''
        self.initialize(k, method)
        ll_prev = 0
        num_iter = 0
        while num_iter < max_iter:
            num_iter += 1 
            self.e_step()
            self.m_step()
            ll = self.log_likelihood()
            self.loglikelihood_hist.append(ll)
            if abs(ll - ll_prev) < stop_tol:
                break
            if animate:
                clear_output(wait=True)
                self.plot_clusters(self.data)
                plt.pause(0.1)
            ll_prev = ll
        print(f'Number of iteration is {num_iter}.')

    def find_outliers(self, thres=0.05):
        '''Find outliers in a dataset using clustering by EM algorithm

        Parameters:
        -----------
        thres: float. Value >= 0
            Outlier defined as data samples assigned to a cluster with probability of belonging to
            that cluster < thres

        Returns:
        -----------
        Python lists of ndarrays. len(Python list) = len(cluster_inds).
            Example if k = 2: [(array([ 0, 17]),), (array([20, 26]),)]
                The Python list has 2 entries. Each entry is a ndarray.
            Within each ndarray, indices of `self.data` of detected outliers according to that cluster.
                For above example: data samples with indices 20 and 26 are outliers according to
                cluster 2.
        '''
        cluster_assig = np.argmax(self.responsibilities, axis=0) 
        result = [] 
        for i in range(self.k):
            indx = np.where(cluster_assig == i)[0]
            data = self.data[indx]
            probs = self.gaussian(data, self.centroids[i], self.cov_mats[i])
            result.append(indx[probs < thres])
        return result
    
    def elbow_plot(self):
        ''' Draw the elbow plot recording the data of log likelihood as data being clustered. 
        '''
        n = len(self.loglikelihood_hist)
        x = np.linspace(1, n, n)
        y = self.loglikelihood_hist.copy()
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set_title('Elbow plot')
        ax.set_xlabel('Number of Iterations')
        ax.set_ylabel('Log Likelihood')

    def estimate_log_probs(self, xy_points):
        '''Used for plotting the clusters.

        (Should not require any changes)
        '''
        probs = np.zeros([self.k, len(xy_points)])
        for c in range(self.k):
            probs[c] = np.log(self.gaussian(xy_points, self.centroids[c], self.cov_mats[c]))
        probs += np.log(self.pi[:, np.newaxis])
        return -logsumexp(probs, axis=0)

    def get_sample_points(self, data, res):
        '''Used for plotting the clusters.

        (Should not require any changes)
        '''
        data_min = np.min(data, axis=0) - 0.5
        data_max = np.max(data, axis=0) + 0.5
        x_samps, y_samps = np.meshgrid(np.linspace(data_min[0], data_max[0], res),
                                       np.linspace(data_min[1], data_max[1], res))
        plt_samps_xy = np.c_[x_samps.ravel(), y_samps.ravel()]
        return plt_samps_xy, x_samps, y_samps

    def plot_clusters(self, data, res=100, show=True):
        '''Method to call to plot the clustering of `data` using the EM algorithm

        (Should not require any changes)
        '''
        # Plot points assigned to each cluster in a different color
        cluster_hard_assignment = np.argmax(self.responsibilities, axis=0)
        for c in range(self.k):
            curr_clust = data[cluster_hard_assignment == c]
            plt.plot(curr_clust[:, 0], curr_clust[:, 1], '.', markersize=7)

        # Plot centroids of each cluster
        plt.plot(self.centroids[:, 0], self.centroids[:, 1], '+k', markersize=12)

        # Get grid of (x,y) points to sample the Gaussian clusters
        xy_points, x_samps, y_samps = self.get_sample_points(data, res=res)

        # Evaluate the sample points at each cluster Gaussian. For visualization, take max prob
        # value of the clusters at each point
        probs = np.zeros([self.k, len(xy_points)])
        for c in range(self.k):
            probs[c] = self.gaussian(xy_points, self.centroids[c], self.cov_mats[c])
        probs /= probs.max(axis=1, keepdims=True)
        probs = probs.sum(axis=0)
        probs = np.reshape(probs, [res, res])

        # Make heatmap for cluster probabilities
        plt.contourf(x_samps, y_samps, probs, cmap='viridis')
        if show:
            plt.show()
