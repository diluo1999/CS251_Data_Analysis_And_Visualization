'''pca_cov.py
Performs principal component analysis using the covariance matrix approach
Curtis Zhuang
CS 251 Data Analysis Visualization, Spring 2020
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class PCA_COV:
    '''
    Perform and store principal component analysis results
    '''

    def __init__(self, data):
        '''

        Parameters:
        -----------
        data: pandas DataFrame. shape=(num_samps, num_vars)
            Contains all the data samples and variables in a dataset.

        (No changes should be needed)
        '''
        self.data = data

        # vars: Python list. len(vars) = num_selected_vars
        #   String variable names selected from the DataFrame to run PCA on.
        #   num_selected_vars <= num_vars
        self.vars = None

        # A: ndarray. shape=(num_samps, num_selected_vars)
        #   Matrix of data selected for PCA
        self.A = None

        # normalized: boolean.
        #   Whether data matrix (A) is normalized by self.pca
        self.normalized = None

        # A_proj: ndarray. shape=(num_samps, num_pcs_to_keep)
        #   Matrix of PCA projected data
        self.A_proj = None

        # orig_means: ndarray. shape=(num_selected_vars,)
        #   Means of each orignal data variable
        self.orig_means = None

        # orig_scales: ndarray. shape=(num_selected_vars,)
        #   Ranges of each orignal data variable
        self.orig_scales = None

        # e_vals: ndarray. shape=(num_pcs,)
        #   Full set of eigenvalues (ordered large-to-small)
        self.e_vals = None
        # e_vecs: ndarray. shape=(num_selected_vars, num_pcs)
        #   Full set of eigenvectors, corresponding to eigenvalues ordered large-to-small
        self.e_vecs = None

        # prop_var: Python list. len(prop_var) = num_pcs
        #   Proportion variance accounted for by the PCs (ordered large-to-small)
        self.prop_var = None

        # cum_var: Python list. len(cum_var) = num_pcs
        #   Cumulative proportion variance accounted for by the PCs (ordered large-to-small)
        self.cum_var = None

    def get_prop_var(self):
        '''(No changes should be needed)'''
        return self.prop_var

    def get_cum_var(self):
        '''(No changes should be needed)'''
        return self.cum_var

    def get_eigenvalues(self):
        '''(No changes should be needed)'''
        return self.e_vals

    def get_eigenvectors(self):
        '''(No changes should be needed)'''
        return self.e_vecs

    def covariance_matrix(self, data):
        '''Computes the covariance matrix of `data`

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_vars)
            `data` is NOT centered coming in, you should do that here.

        Returns:
        -----------
        ndarray. shape=(num_vars, num_vars)
            The covariance matrix of centered `data`

        NOTE: You should do this wihout any loops
        NOTE: np.cov is off-limits here — compute it from "scratch"!
        '''
        mean = np.mean(data, axis=0)

        Ac = data - mean

        cov = 1/(data.shape[0] - 1) * Ac.T @ Ac

        return cov

    def compute_prop_var(self, e_vals):
        '''Computes the proportion variance accounted for by the principal components (PCs).

        Parameters:
        -----------
        e_vals: ndarray. shape=(num_pcs,)

        Returns:
        -----------
        Python list. len = num_pcs
            Proportion variance accounted for by the PCs
        '''

        self.prop_var = []

        sum = np.sum(e_vals)

        for i in e_vals:

            self.prop_var.append(i/sum)

        return self.prop_var

    def compute_cum_var(self, prop_var):
        '''Computes the cumulative variance accounted for by the principal components (PCs).

        Parameters:
        -----------
        prop_var: Python list. len(prop_var) = num_pcs
            Proportion variance accounted for by the PCs, ordered largest-to-smallest
            [Output of self.compute_prop_var()]

        Returns:
        -----------
        Python list. len = num_pcs
            Cumulative variance accounted for by the PCs
        '''

        self.cum_var = []

        cum = 0

        for i in prop_var:

            cum += i

            self.cum_var.append(cum)

        return self.cum_var

    def pca(self, vars, normalize=False):
        '''Performs PCA on the data variables `vars`

        Parameters:
        -----------
        vars: Python list of strings. len(vars) = num_selected_vars
            1+ variable names selected to perform PCA on.
            Variable names must match those used in the `self.data` DataFrame.
        normalize: boolean.
            If True, normalize each data variable so that the values range from 0 to 1.

        NOTE: Leverage other methods in this class as much as possible to do computations.

        TODO:
        - Select the relevant data (corresponding to `vars`) from the data pandas DataFrame
        then convert to numpy ndarray for forthcoming calculations.
        - If `normalize` is True, normalize the selected data so that each variable (column)
        ranges from 0 to 1 (i.e. normalize based on the dynamic range of each variable).
        - Make sure to compute everything needed to set all instance variables defined in constructor,
        except for self.A_proj (this will happen later).
        '''

        # var
        self.vars = vars

        # normalize
        self.normalized = False

        # pandas select columns by variable name
        dat = self.data[vars]

        dat = dat.to_numpy()

        # orig mean
        self.orig_means = np.mean(dat, axis=1)

        # orig scale
        orig_max = np.max(dat, axis=0)

        orig_min = np.min(dat, axis=0)

        self.orig_scales = orig_max - orig_min

        # normalize
        if normalize:
            dat = (dat - orig_min) / self.orig_scales

            self.normalized = True

        # self.A
        self.A = dat

        # calculate eigenvalue and eigenvector

        # eigenvalue, eigenvector
        eigen = np.linalg.eig(self.covariance_matrix(self.A))

        self.e_vals = eigen[0]

        self.e_vecs = eigen[1]

        # prop_var
        self.prop_var = self.compute_prop_var(self.e_vals)

        # cum_var
        self.cum_var = self.compute_cum_var(self.prop_var)


    def elbow_plot(self, num_pcs_to_keep=None):
        '''Plots a curve of the cumulative variance accounted for by the top `num_pcs_to_keep` PCs.
        x axis corresponds to top PCs included (large-to-small order)
        y axis corresponds to proportion variance accounted for

        Parameters:
        -----------
        num_pcs_to_keep: int. Show the variance accounted for by this many top PCs.
            If num_pcs_to_keep is None, show variance accounted for by ALL the PCs (the default).

        NOTE: Make plot markers at each point. Enlarge them so that they look obvious.
        NOTE: Reminder to create useful x and y axis labels.
        NOTE: Don't write plt.show() in this method
        '''

        if num_pcs_to_keep == None:

            num_pcs_to_keep = len(self.prop_var)

        x = []

        y = []

        for i in range(num_pcs_to_keep):

            x.append(i)

            y.append(self.cum_var[i])

        plt.plot(x, y, marker='o')

        plt.xlabel('PCs (large to small)')

        plt.ylabel('Cumulative variance')

        plt.title('Elbow plot for PC')

    def pca_project(self, pcs_to_keep):
        '''Project the data onto `pcs_to_keep` PCs (not necessarily contiguous)

        Parameters:
        -----------
        pcs_to_keep: Python list of ints. len(pcs_to_keep) = num_pcs_to_keep
            Project the data onto these PCs.
            NOTE: This LIST contains indices of PCs to project the data onto, they are NOT necessarily
            contiguous.
            Example 1: [0, 2] would mean project on the 1st and 3rd largest PCs.
            Example 2: [0, 1] would mean project on the two largest PCs.

        Returns
        -----------
        pca_proj: ndarray. shape=(num_samps, num_pcs_to_keep).
            e.g. if pcs_to_keep = [0, 1],
            then pca_proj[:, 0] are x values, pca_proj[:, 1] are y values.

        NOTE: This method should set the variable `self.A_proj`
        '''

        # the remaining eigen vectors (V)
        pcs = self.e_vecs[:, pcs_to_keep]

        # calculate Ac (centered A)

        Ac = self.A - self.A.mean(axis=0)

        self.A_proj = Ac @ pcs

        return self.A_proj

    def loading_plot(self):
        '''Create a loading plot of the top 2 PC eigenvectors

        TODO:
        - Plot a line joining the origin (0, 0) and corresponding components of the top 2 PC eigenvectors.
            Example: If e_1 = [0.1, 0.3] and e_2 = [1.0, 2.0], you would create two lines to join
            (0, 0) and (0.1, 1.0); (0, 0) and (0.3, 2.0).
            Number of lines = num_vars
        - Use plt.annotate to label each line by the variable that it corresponds to.
        - Reminder to create useful x and y axis labels.

        NOTE: Don't write plt.show() in this method
        '''

        pc1 = self.e_vecs[:, 0]

        pc2 = self.e_vecs[:, 1]

        annotate = ['x', 'y', 'z', 'h','p','v', 'w', 'g', 'f','m', 'n']

        for i in range(len(pc2)):

            if i > 10:

                break

            x = pc1[i]

            y = pc2[i]

            plt.plot([0, x], [0, y])

            plt.annotate(annotate[i], xy=(x, y))


        plt.xlabel("PC1")

        plt.ylabel("PC2")


        # how to do this one

    def pca_then_project_back(self, top_k):
        '''Project the data into PCA space (on `top_k` PCs) then project it back to the data space

        Parameters:
        -----------
        top_k: int. Project the data onto this many top PCs.

        Returns:
        -----------
        ndarray. shape=(num_samps, num_selected_vars)

        TODO:
        - Project the data on the `top_k` PCs (assume PCA has already been performed).
        - Project this PCA-transformed data back to the original data space
        - If you normalized, remember to rescale the data projected back to the original data space.
        '''
        top_pcs = []

        for i in range(top_k):

            top_pcs.append(i)

        self.pca_project(top_pcs)

        # Project back to original data space

        pcs = self.e_vecs[:, top_pcs]

        orig_data = (self.A_proj @ pcs.T + self.A.mean(axis = 0)) * self.orig_scales + np.min(self.A, axis=0)

        return orig_data