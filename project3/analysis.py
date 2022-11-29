'''analysis.py
Run statistical analyses and plot Numpy ndarray data
DI LUO
CS 251 Data Analysis Visualization, Spring 2020
'''
import numpy as np
import matplotlib.pyplot as plt


class Analysis:
    def __init__(self, data):
        '''

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        self.data = data

        # Make plot font sizes legible
        plt.rcParams.update({'font.size': 18})

    def set_data(self, data):
        '''Method that re-assigns the instance variable `data` with the parameter.
        Convenience method to change the data used in an analysis without having to create a new
        Analysis object.

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        self.data = data
        return

    def min(self, headers, rows=[]):
        '''Computes the minimum of each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.
        (i.e. the minimum value in each of the selected columns)

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of min over, or over all indices
            if rows=[]

        Returns
        -----------
        mins: ndarray. shape=(len(headers),)
            Minimum values for each of the selected header variables

        NOTE: Loops are forbidden!
        '''

        return np.min(self.data.select_data(headers, rows), axis=0)

    def max(self, headers, rows=[]):
        '''Computes the maximum of each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of max over, or over all indices
            if rows=[]

        Returns
        -----------
        maxs: ndarray. shape=(len(headers),)
            Maximum values for each of the selected header variables

        NOTE: Loops are forbidden!
        '''

        return np.max(self.data.select_data(headers, rows), axis=0)

    def range(self, headers, rows=[]):
        '''Computes the range [min, max] for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of min/max over, or over all indices
            if rows=[]

        Returns
        -----------
        mins: ndarray. shape=(len(headers),)
            Minimum values for each of the selected header variables
        maxes: ndarray. shape=(len(headers),)
            Maximum values for each of the selected header variables

        NOTE: Loops are forbidden!
        '''
        mins = self.min(headers, rows)
        maxs = self.max(headers, rows)
        return mins, maxs

    def mean(self, headers, rows=[]):
        '''Computes the mean for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`).

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of mean over, or over all indices
            if rows=[]

        Returns
        -----------
        means: ndarray. shape=(len(headers),)
            Mean values for each of the selected header variables

        NOTE: You CANNOT use np.mean here!
        NOTE: Loops are forbidden!
        '''

        if len(rows) == 0:
            return (np.sum(self.data.select_data(headers, rows), axis=0))/self.data.get_num_samples()
        else:
            return (np.sum(self.data.select_data(headers, rows), axis=0))/len(rows)

    def var(self, headers, rows=[]):
        '''Computes the variance for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of variance over, or over all indices
            if rows=[]

        Returns
        -----------
        vars: ndarray. shape=(len(headers),)
            Variance values for each of the selected header variables

        NOTE: You CANNOT use np.var or np.mean here!
        NOTE: Loops are forbidden!
        '''
        
        if len(rows) == 0:
            return np.sum(np.square(self.data.select_data(headers, rows)-self.mean(headers, rows)), axis=0)/(self.data.get_num_samples()-1)
        else:
            return np.sum(np.square(self.data.select_data(headers, rows)-self.mean(headers, rows)), axis=0)/(len(rows)-1)


    def std(self, headers, rows=[]):
        '''Computes the standard deviation for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of standard deviation over,
            or over all indices if rows=[]

        Returns
        -----------
        vars: ndarray. shape=(len(headers),)
            Standard deviation values for each of the selected header variables

        NOTE: You CANNOT use np.var, np.std, or np.mean here!
        NOTE: Loops are forbidden!
        '''
        return np.sqrt(self.var(headers, rows))

    # Proj2 Extension
    def median(self, headers, rows=[]):
        '''Computes the median for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of median over,
            or over all indices if rows=[]

        Returns
        -----------
        vars: ndarray. shape=(len(headers),)
            Median values for each of the selected header variables
        '''
        return np.median(self.data.select_data(headers, rows), axis=0)
    
    # Proj2 Extension
    def quartile(self, q, headers, rows=[]):
        '''Computes the 1st or 3rd quartile for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        q: int
            A int indicating whether 1st or 3rd quartile is calculated
            1 indicates 1st quartile
            3 indicates 3rd quartile
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of 1st/3rd quartile over,
            or over all indices if rows=[]

        Returns
        -----------
        None if wrong number is used to indicated 1st/3rd quartile
        vars: ndarray. shape=(len(headers),)
            1st/3rd quartile values for each of the selected header variables
        '''

        if q == 1 or q == 3:
            percentile = 25*q
        else:
            print('The first parameter should be 1 or 3, in which 1 and 3 indicate 1st and 3rd quartile respectively.')
            return None

        return np.percentile(self.data.select_data(headers, rows), q=percentile, axis=0)


    def show(self):
        '''Simple wrapper function for matplotlib's show function.

        (Does not require modification)
        '''
        plt.show()

    def scatter(self, ind_var, dep_var, title):
        '''Creates a simple scatter plot with "x" variable in the dataset `ind_var` and
        "y" variable in the dataset `dep_var`. Both `ind_var` and `dep_var` should be strings
        in `self.headers`.

        Parameters:
        -----------
        ind_var: str.
            Name of variable that is plotted along the x axis
        dep_var: str.
            Name of variable that is plotted along the y axis
        title: str.
            Title of the scatter plot

        Returns:
        -----------
        x. ndarray. shape=(num_data_samps,)
            The x values that appear in the scatter plot
        y. ndarray. shape=(num_data_samps,)
            The y values that appear in the scatter plot

        NOTE: Do not call plt.show() here.
        '''
        x = self.data.select_data(ind_var).reshape(self.data.get_num_samples(),)
        y = self.data.select_data(dep_var).reshape(self.data.get_num_samples(),)
        plt.scatter(x, y)
        plt.title(title)
        plt.xlabel(ind_var)
        plt.ylabel(dep_var)
        return x, y

    # Proj2 Extension
    def plot(self, ind_var, dep_var, title):
        '''Creates a simple line plot with "x" variable in the dataset `ind_var` and
        "y" variable in the dataset `dep_var`. Both `ind_var` and `dep_var` should be strings
        in `self.headers`.

        Parameters:
        -----------
        ind_var: str.
            Name of variable that is plotted along the x axis
        dep_var: str.
            Name of variable that is plotted along the y axis
        title: str.
            Title of the line plot

        Returns:
        -----------
        x. ndarray. shape=(num_data_samps,)
            The x values that appear in the line plot
        y. ndarray. shape=(num_data_samps,)
            The y values that appear in the line plot
        '''
        x = self.data.select_data(ind_var).reshape(self.data.get_num_samples(),)
        y = self.data.select_data(dep_var).reshape(self.data.get_num_samples(),)
        plt.plot(x, y)
        plt.title(title)
        return x, y
    
    # Proj2 Extension
    def hist(self, ind_var, title):
        '''Creates a simple histogram with "x" variable in the dataset `ind_var`.
        `ind_var` should be strings in `self.headers`.

        Parameters:
        -----------
        ind_var: str.
            Name of variable that is plotted along the x axis
        title: str.
            Title of the histogram

        Returns:
        -----------
        x. ndarray. shape=(num_data_samps,)
            The x values that appear in the histogram
        '''
        x = self.data.select_data(ind_var).reshape(self.data.get_num_samples(),)
        plt.hist(x)
        plt.title(title)
        return x
    
    # Proj2 Extension
    def hist_r(self, ind_var, r, bin=100, title='Histogram', x_axis='X'):
        '''Creates a histogram with a range of "x" variable in the dataset `ind_var`.
        `ind_var` should be strings in `self.headers`, `r` should be a ndarray that
        includes indices of rows that want to be selected to plot.

        Parameters:
        -----------
        ind_var: str.
            Name of variable that is plotted along the x axis
        r: ndarry
            array of indices of rows that want to be selected to plot
        bin: int
        title: str.
            Title of the histogram
        x_axis: str

        Returns:
        -----------
        x. ndarray. shape=(num_data_samps,)
            The x values that appear in the histogram
        '''
        x = self.data.select_data(ind_var, r).reshape(len(r),)
        plt.hist(x, bins=bin)
        plt.title(title)
        plt.xlabel(x_axis)
        plt.ylabel('frequency')
        return x
    
    # Proj2 Extension
    def bar(self, ind_var, title='', tick_label='', ylabel=''):
        '''Creates a bar plot with the dataset `ind_var`.

        Parameters:
        -----------
        ind_var: str.
            Name of variable that is plotted along the x axis
        title: str.
            Title of the histogram
        tick_label: list of str
            Labels for tick
        ylabel: str

        '''
        x = np.arange(len(ind_var))
        width = 0.35
        rects = plt.bar(x,ind_var, tick_label=tick_label)
        plt.title(title)
        plt.ylabel(ylabel)

        for rect in rects:
            height = rect.get_height()
            plt.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    def pair_plot(self, data_vars, fig_sz=(12, 12), title=''):
        '''Create a pair plot: grid of scatter plots showing all combinations of variables in
        `data_vars` in the x and y axes.

        Parameters:
        -----------
        data_vars: Python list of str.
            Variables to place on either the x or y axis of the scatter plots
        fig_sz: tuple of 2 ints.
            The width and height of the figure of subplots. Pass as a paramter to plt.subplots.
        title. str. Title for entire figure (not the individual subplots)

        Returns:
        -----------
        fig. The matplotlib figure.
            1st item returned by plt.subplots
        axes. ndarray of AxesSubplot objects. shape=(len(data_vars), len(data_vars))
            2nd item returned by plt.subplots

        TODO:
        - Make the len(data_vars) x len(data_vars) grid of scatterplots
        - The y axis of the first column should be labeled with the appropriate variable being
        plotted there.
        - The x axis of the last row should be labeled with the appropriate variable being plotted
        there.
        - There should be no other axis or tick labels (it looks too cluttered otherwise!)

        Tip: Check out the sharex and sharey optional parameters of plt.subplots
        '''
        M=len(data_vars)
        selected_variables = self.data.select_data(data_vars)
        fig, axes = plt.subplots(nrows=M, ncols=M, sharex='all', sharey='all', figsize=fig_sz)
        fig.suptitle(title)

        for i in range(M):
            for j in range(M):
                axes[i, j].scatter(selected_variables[:,j], selected_variables[:,i])
                if i==M-1:
                    axes[i, j].set_xlabel(data_vars[j])
                if j==0:
                    axes[i, j].set_ylabel(data_vars[i])
        return fig, axes
