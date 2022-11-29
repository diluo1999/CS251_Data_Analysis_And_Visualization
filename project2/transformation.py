'''transformation.py
Perform projections, translations, rotations, and scaling operations on Numpy ndarray data.
DI LUO
CS 251 Data Analysis Visualization, Spring 2020
'''
import numpy as np
import matplotlib.pyplot as plt
import analysis
import data
from palettable.colorbrewer.sequential import Greys_9
from mpl_toolkits.mplot3d import Axes3D


class Transformation(analysis.Analysis):

    def __init__(self, data_orig, data=None):
        '''Constructor for a Transformation object

        Parameters:
        -----------
        data_orig: Data object. shape=(N, num_vars).
            Contains the original dataset (only containing all the numeric variables
            — `num_vars` in total).
        data: Data object (or None). shape=(N, num_proj_vars).
            Contains all the data samples as the original, but ONLY A SUBSET of the variables.
            (`num_proj_vars` in total). `num_proj_vars` <= `num_vars`

        TODO:
        - Pass `data` to the superclass constructor.
        - Create an instance variables for `data_orig`.
        '''

        analysis.Analysis.__init__(self, data=data)
        self.data_orig = data_orig

    def project(self, headers):
        '''Project the data on the list of data variables specified by `headers` — i.e. select a
        subset of the variables from the original dataset. In other words, populate the instance
        variable `self.data`.

        Parameters:
        -----------
        headers: Python list of str. len(headers) = `num_proj_vars`, usually 1-3 (inclusive), but
            there could be more.
            A list of headers (strings) specifying the feature to be projected onto each axis.
            For example: if headers = ['hi', 'there', 'cs251'], then the data variables
                'hi' becomes the 'x' variable,
                'there' becomes the 'y' variable,
                'cs251' becomes the 'z' variable.
            The length of the list dictates the number of dimensions onto which the dataset is
            projected — having 'y' and 'z' variables are optional.

        HINT: Update self.data with a new Data object and fill in appropriate optional parameters
        (except for `filepath`)

        TODO:
        - Create a new `Data` object that you assign to `self.data` (project data onto the `headers`
        variables).
        - Make sure that you create 'valid' values for all the `Data` constructor optional parameters
        (except you dont need `filepath` because it is not relevant).
        '''
        
        # Project the data on the list of data variables specified by `headers` with data.select_data()
        data_new = self.data_orig.select_data(headers)

        # Create new Python dictionary that maps header (var str name) to column index (int)
        header2col_new = {}
        index = 0
        for i in headers:
            header2col_new[i] = index
            index += 1
        
        # Update self.data with a new Data object and fill in appropriate optional parameters
        self.data = data.Data(headers=headers, data=data_new, header2col=header2col_new)

    def get_data_homogeneous(self):
        '''Helper method to get a version of the projected data array with an added homogeneous
        coordinate. Useful for homogeneous transformations.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars+1). The projected data array with an added 'fake variable'
        column of ones on the right-hand side.
            For example: If we have the data SAMPLE (just one row) in the projected data array:
            [3.3, 5.0, 2.0], this sample would become [3.3, 5.0, 2.0, 1] in the returned array.

        NOTE:
        - Do NOT update self.data with the homogenous coordinate.
        '''

        # Create homogeneous_coordinate
        homogeneous_coordinate = np.ones([self.data.get_num_samples(),1])

        # Get a copy of all the data in Data object
        data_copy = self.data.get_all_data()

        # Return the projected data array with an added homogeneous coordinate with np.hstack()
        return np.hstack((data_copy, homogeneous_coordinate))



    def translation_matrix(self, headers, magnitudes):
        ''' Make an M-dimensional homogeneous transformation matrix for translation,
        where M is the number of features in the projected dataset.

        Parameters:
        -----------
        headers: Python list of str.
            Specifies the variables along which the projected dataset should be translated.
        magnitudes: Python list of float.
            Translate corresponding variables in `headers` (in the projected dataset) by these
            amounts.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars+1). The transformation matrix.

        NOTE: This method just creates the translation matrix. It does NOT actually PERFORM the
        translation!
        '''
        
        # Create the transformation matrix
        M = self.data.get_num_dims()
        t_matrix = np.eye(M+1)

        # Specify the amount of translating for correponding variables 
        indList = self.data.get_header_indices(headers)
        index = 0
        for i in indList:
            t_matrix[i, M] = magnitudes[index]
            index += 1

        return t_matrix

    def scale_matrix(self, headers, magnitudes):
        '''Make an M-dimensional homogeneous scaling matrix for scaling, where M is the number of
        variables in the projected dataset.

        Parameters:
        -----------
        headers: Python list of str.
            Specifies the variables along which the projected dataset should be scaled.
        magnitudes: Python list of float.
            Scale corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars+1). The scaling matrix.

        NOTE: This method just creates the scaling matrix. It does NOT actually PERFORM the scaling!
        '''
        
        # Create the scaling matrix
        M = self.data.get_num_dims()
        s_matrix = np.eye(M+1)

        # Specify the amount of scaling for correponding variables 
        indList = self.data.get_header_indices(headers)
        index = 0
        for i in indList:
            s_matrix[i, i] = magnitudes[index]
            index += 1
        
        return s_matrix

    def rotation_matrix_3d(self, header, degrees):
        '''Make an 3-D homogeneous rotation matrix for rotating the projected data about the ONE
        axis/variable `header`.

        Parameters:
        -----------
        header: str. Specifies the variable about which the projected dataset should be rotated.
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(4, 4). The 3D rotation matrix with homogenous coordinate.

        NOTE: This method just creates the rotation matrix. It does NOT actually PERFORM the rotation!
        '''

        # Create the rotation matrix
        r_matrix = np.eye(4)

        # Find the variable about which the projected dataset should be rotated
        axis = self.data.get_mappings()[header]

        # Calculate cos() and sin()
        cos = np.cos(degrees*np.pi/180)
        sin = np.sin(degrees*np.pi/180)
        
        if axis == 0:
            r_matrix[1,1] = cos
            r_matrix[1,2] = -sin
            r_matrix[2,1] = sin
            r_matrix[2,2] = cos
        elif axis == 1:
            r_matrix[0,0] = cos
            r_matrix[0,2] = sin
            r_matrix[2,0] = -sin
            r_matrix[2,2] = cos
        elif axis == 2:
            r_matrix[0,0] = cos
            r_matrix[0,1] = -sin
            r_matrix[1,0] = sin
            r_matrix[1,1] = cos
        else:
            print('Invalid variable. Please check the parameter `headers`.')
            return

        return r_matrix
    
    # Extension 3: Implement and use 2D rotation
    def rotation_matrix_2d(self, degrees):
        '''Make an 2-D homogeneous rotation matrix for rotating the projected data.

        Parameters:
        -----------
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(3, 3). The 2D rotation matrix with homogenous coordinate.
        '''

        # Create the rotation matrix
        r_matrix = np.eye(3)

        # Calculate cos() and sin()
        cos = np.cos(degrees*np.pi/180)
        sin = np.sin(degrees*np.pi/180)
        
        r_matrix[0,0] = cos
        r_matrix[0,1] = -sin
        r_matrix[1,0] = sin
        r_matrix[1,1] = cos

        return r_matrix


    def transform(self, C):
        '''Transforms the PROJECTED dataset by applying the homogeneous transformation matrix `C`.

        Parameters:
        -----------
        C: ndarray. shape=(num_proj_vars+1, num_proj_vars+1).
            A homogeneous transformation matrix.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars+1). The projected dataset after it has been transformed by `C`
        '''

        A = self.get_data_homogeneous()
        return (C @ A.T).T

    def translate(self, headers, magnitudes):
        '''Translates the variables `headers` in projected dataset in corresponding amounts specified
        by `magnitudes`.

        Parameters:
        -----------
        headers: Python list of str.
            Specifies the variables along which the projected dataset should be translated.
        magnitudes: Python list of float.
            Translate corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The translated data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplcation to translate the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a homogenous
        coordinate!
        '''
        t = self.translation_matrix(headers, magnitudes)
        new_data = self.transform(t)[:, :-1]
        self.data = data.Data(headers=self.data.get_headers(), data=new_data, header2col=self.data.get_mappings())

        return new_data

    def scale(self, headers, magnitudes):
        '''Scales the variables `headers` in projected dataset in corresponding amounts specified
        by `magnitudes`.

        Parameters:
        -----------
        headers: Python list of str.
            Specifies the variables along which the projected dataset should be scaled.
        magnitudes: Python list of float.
            Scale corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The scaled data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplcation to scale the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a
        homogenous coordinate!
        '''
        s = self.scale_matrix(headers, magnitudes)
        new_data = self.transform(s)[:, :-1]
        self.data = data.Data(headers=self.data.get_headers(), data=new_data, header2col=self.data.get_mappings())

        return new_data

    def rotate_3d(self, header, degrees):
        '''Rotates the projected data about the variable `header` by the angle (in degrees)
        `degrees`.

        Parameters:
        -----------
        header: str. Specifies the variable about which the projected dataset should be rotated.
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The rotated data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplcation to rotate the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a
        homogenous coordinate!
        '''
        r = self.rotation_matrix_3d(header, degrees)
        new_data = self.transform(r)[:, :-1]
        self.data = data.Data(headers=self.data.get_headers(), data=new_data, header2col=self.data.get_mappings())

        return new_data
    
    # Extension 3: Implement and use 2D rotation
    def rotate_2d(self, degrees):
        '''Rotates the projected data by the angle (in degrees) `degrees` in 2D.

        Parameters:
        -----------
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The rotated data (with all variables in the projected).
            dataset.
        '''

        r = self.rotation_matrix_2d(degrees)
        new_data = self.transform(r)[:, :-1]
        self.data = data.Data(headers=self.data.get_headers(), data=new_data, header2col=self.data.get_mappings())

        return new_data

    def normalize_together(self):
        '''Normalize all variables in the projected dataset together by translating the global minimum
        (across all variables) to zero and scaling the global range (across all variables) to one.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.
        '''
        
        gloMax = np.max(self.data.data)
        gloMin = np.min(self.data.data)
        gloRange = gloMax - gloMin

        headers = self.data.get_headers()
        mag1 = [-gloMin for i in range(len(headers))]
        translated_matrix = self.translate(headers, mag1)

        mag2 = [(1/gloRange) for i in range(len(headers))]
        scaled_matrix = self.scale(headers, mag2)

        return scaled_matrix

    # Extension 2: Implement normalize together and separately using numpy vectorization/broadcasting. 
    # Compare the approaches in efficiency (time and compare the two implementations)
    def normalize_together_vectorization(self):
        '''Normalize all variables in the projected dataset together by numpy vectorization/broadcasting.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.
        '''
        
        gloMax = np.max(self.data.data)
        gloMin = np.min(self.data.data)
        gloRange = gloMax - gloMin

        n_matrix = (self.data.data - gloMin)/gloRange
        self.data.data = n_matrix

        return n_matrix

    def normalize_separately(self):
        '''Normalize each variable separately by translating its local minimum to zero and scaling
        its local range to one.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.
        '''
        
        headers = self.data.get_headers()
        mins, maxs = self.range(headers)
        ranges = maxs - mins

        translated_matrix = self.translate(headers, -mins)
        scaled_matrix = self.scale(headers, 1/ranges)

        return scaled_matrix
    
    # Extension 2: Implement normalize together and separately using numpy vectorization/broadcasting. 
    # Compare the approaches in efficiency (time and compare the two implementations)
    def normalize_separately_vectorization(self):
        '''Normalize each variable separately by numpy vectorization/broadcasting.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.
        '''
        
        headers = self.data.get_headers()
        mins, maxs = self.range(headers)
        ranges = maxs - mins

        n_matrix = (self.data.data - mins)/ranges
        self.data.data = n_matrix

        return n_matrix

    def scatter_color(self, ind_var, dep_var, c_var, title=None):
        '''Creates a 2D scatter plot with a color scale representing the 3rd dimension.

        Parameters:
        -----------
        ind_var: str. Header of the variable that will be plotted along the X axis.
        dep_var: Header of the variable that will be plotted along the Y axis.
        c_var: Header of the variable that will be plotted along the color axis.
            NOTE: Section B (Linear Algebra): Use a ColorBrewer color palette (e.g. from the
            `palettable` library).
        title: str or None. Optional title that will appear at the top of the figure.
        '''

        x = self.data.select_data(ind_var).reshape(self.data.get_num_samples(),)
        y = self.data.select_data(dep_var).reshape(self.data.get_num_samples(),)
        z = self.data.select_data(c_var).reshape(self.data.get_num_samples(),)

        fig, ax = plt.subplots()

        # Correctly use cmap: Credit to Zixuan Wang
        scatter = ax.scatter(x, y, c=z, cmap=Greys_9.mpl_colormap, edgecolors='gray')
        
        if title is not None:
            plt.title(title)

        ax.set_xlabel(ind_var)
        ax.set_ylabel(dep_var)

        # Create and label the colorbar
        cbar = fig.colorbar(scatter)
        cbar.ax.set_ylabel(c_var)

    # Extension 1:  Explore additional visualizations
    def scatter_size(self, ind_var, dep_var, s_var, title=None):
        '''Creates a 2D scatter plot with a marker size scale representing the 3rd dimension.

        Parameters:
        -----------
        ind_var: str. Header of the variable that will be plotted along the X axis.
        dep_var: Header of the variable that will be plotted along the Y axis.
        s_var: Header of the variable that will be plotted along the size axis.
        title: str or None. Optional title that will appear at the top of the figure.
        '''

        x = self.data.select_data(ind_var).reshape(self.data.get_num_samples(),)
        y = self.data.select_data(dep_var).reshape(self.data.get_num_samples(),)
        z = self.data.select_data(s_var).reshape(self.data.get_num_samples(),)

        fig, ax = plt.subplots()

        scatter = ax.scatter(x, y, s=z*20)
        
        if title is not None:
            plt.title(title)

        ax.set_xlabel(ind_var)
        ax.set_ylabel(dep_var)
    
    # Extension 1:  Explore additional visualizations
    def scatter_colorNsize(self, x_var, y_var, z_var, c_var, s_var, title=None):
        '''Creates a 5D scatter plot with a color scale representing the 4the dimension
        and marker size representing the 5th dimension.

        Parameters:
        -----------
        x_var: str. Header of the variable that will be plotted along the X axis.
        y_var: Header of the variable that will be plotted along the Y axis.
        z_var: Header of the variable that will be plotted along the Z axis.
        c_var: Header of the variable that will be plotted along the color axis.
        s_var: Header of the variable that will be plotted along the size axis.
        title: str or None. Optional title that will appear at the top of the figure.
        '''

        x = self.data.select_data(x_var).reshape(self.data.get_num_samples(),)
        y = self.data.select_data(y_var).reshape(self.data.get_num_samples(),)
        z = self.data.select_data(z_var).reshape(self.data.get_num_samples(),)
        c = self.data.select_data(c_var).reshape(self.data.get_num_samples(),)
        s = self.data.select_data(s_var).reshape(self.data.get_num_samples(),)

        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(xs=x, ys=y, zs=z, s=s*20, c=c, cmap=Greys_9.mpl_colormap)
        
        if title is not None:
            ax.set_title(title)

        ax.set_xlabel(x_var)
        ax.set_ylabel(y_var)
        ax.set_zlabel(z_var)

        # Create and label the colorbar
        cbar = fig.colorbar(scatter)
        cbar.ax.set_ylabel(c_var)


    def heatmap(self, headers=None, title=None, cmap="gray"):
        '''Generates a heatmap of the specified variables (defaults to all). Each variable is normalized
        separately and represented as its own row. Each individual is represented as its own column.
        Normalizing each variable separately means that one color axis can be used to represent all
        variables, 0.0 to 1.0.

        Parameters:
        -----------
        headers: Python list of str (or None). (Optional) The variables to include in the heatmap.
            Defaults to all variables if no list provided.
        title: str. (Optional) The figure title. Defaults to an empty string (no title will be displayed).
        cmap: str. The colormap string to apply to the heatmap. Defaults to grayscale
            -- black (0.0) to white (1.0)

        Returns:
        -----------
        fig, ax: references to the figure and axes on which the heatmap has been plotted
        '''

        # Create a doppelganger of this Transformation object so that self.data
        # remains unmodified when heatmap is done
        data_clone = data.Data(headers=self.data.get_headers(),
                               data=self.data.get_all_data(),
                               header2col=self.data.get_mappings())
        dopp = Transformation(self.data, data_clone)
        dopp.normalize_separately()

        fig, ax = plt.subplots()
        if title is not None:
            ax.set_title(title)
        ax.set(xlabel="Individuals")

        # Select features to plot
        if headers is None:
            headers = dopp.data.headers
        m = dopp.data.select_data(headers)

        # Generate heatmap
        hmap = ax.imshow(m.T, aspect="auto", cmap=cmap)

        # Label the features (rows) along the Y axis
        y_lbl_coords = np.arange(m.shape[1]+1) - 0.5
        ax.set_yticks(y_lbl_coords, minor=True)
        y_lbls = [""] + headers
        ax.set_yticklabels(y_lbls )
        ax.grid(linestyle='none')

        # Create and label the colorbar
        cbar = fig.colorbar(hmap)
        cbar.ax.set_ylabel("Normalized Features")

        return fig, ax
