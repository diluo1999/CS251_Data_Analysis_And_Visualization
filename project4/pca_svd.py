'''pca_svd.py
Subclass of PCA_COV that performs PCA using the singular value decomposition (SVD)
Di Luo
CS 251 Data Analysis Visualization, Spring 2020
'''
import numpy as np

import pca_cov


class PCA_SVD(pca_cov.PCA_COV):
    def pca(self, vars, normalize=False):
        '''Performs PCA on the data variables `vars` using SVD

        Parameters:
        -----------
        vars: Python list of strings. len(vars) = num_selected_vars
            1+ variable names selected to perform PCA on.
            Variable names must match those used in the `self.data` DataFrame.
        normalize: boolean.
            If True, normalize each data variable so that the values range from 0 to 1.

        TODO:
        - This method should mirror that in pca_cov.py (same instance variables variables need to
        be computed).
        - There should NOT be any covariance matrix calculation here!
        - You may use np.linalg.svd to perform the singular value decomposition.
        '''
        self.vars = vars
        self.A = self.data[vars].to_numpy()
        self.normalized = normalize

        self.orig_mins = np.min(self.A, axis=0)
        max = np.max(self.A, axis=0)
        self.orig_scales = max - self.orig_mins
        if self.normalized:
            self.A = ( self.A - self.orig_mins ) / self.orig_scales

        self.orig_means = np.mean(self.A, axis=0)
        Ac = self.A -  self.orig_means
        U, S, V = np.linalg.svd(Ac)
        self.e_vals = np.square(S)/(self.A.shape[0]-1)
        self.e_vecs = V.T

        self.prop_var = self.compute_prop_var(self.e_vals)
        self.cum_var = self.compute_cum_var(self.prop_var)
