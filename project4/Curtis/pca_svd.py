'''pca_svd.py
Subclass of PCA_COV that performs PCA using the singular value decomposition (SVD)
YOUR NAME HERE
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

        # 1. select center data
        self.vars = vars

        dat = self.data[vars]

        dat = dat.to_numpy()

        # normalized data
        orig_max = np.max(dat, axis=0)

        orig_min = np.min(dat, axis=0)

        self.orig_scales = orig_max - orig_min

        if normalize:

            dat = (dat - orig_min)/self.orig_scales

            self.normalized = True

        # center data
        self.A = dat - dat.mean(axis = 0)

        # 2. Compute Ac = USVt

        U, S, V = np.linalg.svd(self.A)

        self.e_vecs = V.T

        self.e_vals = np.power(S,2)/(len(self.A) - 1)

        self.prop_var = self.compute_prop_var(self.e_vals)

        self.cum_var = self.compute_cum_var(self.prop_var)