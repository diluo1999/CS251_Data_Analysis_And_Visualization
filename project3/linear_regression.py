'''linear_regression.py
Subclass of Analysis that performs linear regression on data
Di Luo
CS 251 Data Analysis Visualization, Spring 2020
'''
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

import analysis


class LinearRegression(analysis.Analysis):
    '''
    Perform and store linear regression and related analyses
    '''

    def __init__(self, data):
        '''

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        super().__init__(data)

        # ind_vars: Python list of strings.
        #   1+ Independent variables (predictors) entered in the regression.
        self.ind_vars = None
        # dep_var: string. Dependent variable predicted by the regression.
        self.dep_var = None

        # A: ndarray. shape=(num_data_samps, num_ind_vars)
        #   Matrix for independent (predictor) variables in linear regression
        self.A = None

        # y: ndarray. shape=(num_data_samps, 1)
        #   Vector for dependent variable predictions from linear regression
        self.y = None

        # R2: float. R^2 statistic
        self.R2 = None

        # slope: ndarray. shape=(num_ind_vars, 1)
        #   Regression slope(s)
        self.slope = None
        # intercept: float. Regression intercept
        self.intercept = None
        # residuals: ndarray. shape=(num_data_samps, 1)
        #   Residuals from regression fit
        self.residuals = None

        # p: int. Polynomial degree of regression model (Week 2)
        self.p = 1

    def linear_regression(self, ind_vars, dep_var, method='scipy'):
        '''Performs a linear regression on the independent (predictor) variable(s) `ind_vars`
        and dependent variable `dep_var` using the method `method`.

        Parameters:
        -----------
        ind_vars: Python list of strings. 1+ independent variables (predictors) entered in the regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. 1 dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.
        method: str. Method used to compute the linear regression. Here are the options:
            'scipy': Use scipy's linregress function.
            'normal': Use normal equations.
            'qr': Use QR factorization (linear algebra section only).

        TODO:
        - Use your data object to select the variable columns associated with the independent and
        dependent variable strings.
        - Perform linear regression using the appropriate method.
        - Compute R^2 on the fit and the residuals.
        - By the end of this method, all instance variables should be set (see constructor), except
        for self.adj_R2.

        NOTE: Use other methods in this class where ever possible (do not write the same code twice!)
        '''

        # if ind_vars in self.data.get_headers():
        #     self.ind_vars = ind_vars
        # else:
        #     print("Invalid independent variable")
        #     return
        # if dep_var in self.data.get_headers():
        #     self.dep_var = dep_var
        # else:
        #     print("Invalid dependent variable")
        #     return
        self.ind_vars = ind_vars
        self.dep_var = dep_var

        # Find independent variables and dependent variables
        self.A = self.data.select_data(self.ind_vars)
        self.y = self.data.select_data(self.dep_var)

        # Perform linear regression
        if method == 'scipy':
            c = self.linear_regression_scipy(self.A, self.y)
        elif method == 'normal':
            c = self.linear_regression_normal(self.A, self.y)
        elif method == 'qr':
            c = self.linear_regression_qr(self.A, self.y)
        else:
            print('Invalid method')
            return

        self.slope = c[:-1].reshape(len(c)-1,1)
        self.intercept = float(c[-1])

        y_pred = self.predict(self.slope, self.intercept)

        # compute R^2 on the fit and the residuals
        self.residuals = self.compute_residuals(y_pred)
        self.R2 = self.r_squared(y_pred)

    def linear_regression_scipy(self, A, y):
        '''Performs a linear regression using scipy's built-in least squares solver (scipy.linalg.lstsq).
        Solves the equation y = Ac for the coefficient vector c.

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_ind_vars).
            Data matrix for independent variables.
        y: ndarray. shape=(num_data_samps, 1).
            Data column for dependent variable.

        Returns
        -----------
        c: ndarray. shape=(num_ind_vars+1,)
            Linear regression slope coefficients for each independent var PLUS the intercept term
        '''

        N = A.shape[0]
        homo = np.ones([N,1])
        homo_A = np.hstack((A, homo))
        c, res, rnk, s = scipy.linalg.lstsq(homo_A, y)

        return c

    def linear_regression_normal(self, A, y):
        '''Performs a linear regression using the normal equations.
        Solves the equation y = Ac for the coefficient vector c.

        See notebook for a refresher on the equation

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_ind_vars).
            Data matrix for independent variables.
        y: ndarray. shape=(num_data_samps, 1).
            Data column for dependent variable.

        Returns
        -----------
        c: ndarray. shape=(num_ind_vars+1,)
            Linear regression slope coefficients for each independent var AND the intercept term
        '''
        
        N = A.shape[0]
        homo = np.ones([N,1])
        homo_A = np.hstack((A, homo))
        c = np.linalg.inv(homo_A.T @ homo_A) @ homo_A.T @ y

        return c

    def linear_regression_qr(self, A, y):
        '''Performs a linear regression using the QR decomposition

        (Week 2)

        See notebook for a refresher on the equation

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_ind_vars).
            Data matrix for independent variables.
        y: ndarray. shape=(num_data_samps, 1).
            Data column for dependent variable.

        Returns
        -----------
        c: ndarray. shape=(num_ind_vars+1,)
            Linear regression slope coefficients for each independent var AND the intercept term

        NOTE: You should not compute any matrix inverses! Check out scipy.linalg.solve_triangular
        to backsubsitute to solve for the regression coefficients `c`.
        '''
        
        # Add homogeneous coordinates to A
        N = A.shape[0]
        homo = np.ones([N,1])
        homo_A = np.hstack((A, homo))

        # Compute Q and R with qr_decomposition
        Q, R = self.qr_decomposition(homo_A)

        # Solve for c without inverse by using scipy.linalg.solve_triangular
        c = scipy.linalg.solve_triangular(R, Q.T @ y)
        
        return c

    def qr_decomposition(self, A):
        '''Performs a QR decomposition on the matrix A. Make column vectors orthogonal relative
        to each other. Uses the Gram–Schmidt algorithm

        (Week 2)

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_ind_vars+1).
            Data matrix for independent variables.
        
        Returns:
        -----------
        Q: ndarray. shape=(num_data_samps, num_ind_vars+1)
            Orthonormal matrix (columns are orthogonal unit vectors — i.e. length = 1)
        R: ndarray. shape=(num_ind_vars+1, num_ind_vars+1)
            Upper triangular matrix

        TODO:
        - Q is found by the Gram–Schmidt orthogonalizing algorithm.
        Summary: Step thru columns of A left-to-right. You are making each newly visited column
        orthogonal to all the previous ones. You do this by projecting the current column onto each
        of the previous ones and subtracting each projection from the current column.
            - NOTE: Very important: Make sure that you make a COPY of your current column before
            subtracting (otherwise you might modify data in A!).
        Normalize each current column after orthogonalizing.
        - R is found by equation summarized in notebook
        '''

        M = A.shape[1]
        N = A.shape[0]

        # Initializa Q
        Q = np.ones([N, M])

        # Compute Q
        # Visit each column i of A to make orthogonal
        for i in range(M):
            col_i = np.copy(A[:, i])
            # Visit each orthogonized column j and subtract column i by its projection on column j
            for j in range(i):
                col_j = Q[:, j]
                col_i = col_i - np.dot(col_j, col_i) * col_j
            col_i = col_i / np.linalg.norm(col_i)
            Q[:, i] = col_i
        
        # Compute R
        R = Q.T @ A

        return Q, R

    def predict(self, slope, intercept, X=None):
        '''Use fitted linear regression model to predict the values of data matrix `X`.
        Generates the predictions y_pred = mD + b, where (m, b) are the model fit slope and intercept,
        D is the data matrix.

        Parameters:
        -----------
        slope: ndarray. shape=(num_ind_vars, 1)
            Slope coefficients for the linear regression fits for each independent var
        intercept: float.
            Intercept for the linear regression fit
        X: ndarray. shape=(num_data_samps, num_ind_vars).
            If None, use self.A for the "x values" when making predictions.
            If not None, use X as independent var data as "x values" used in making predictions.
        
        Returns
        -----------
        y_pred: ndarray. shape=(num_data_samps,)
            Predicted y (dependent variable) values

        NOTE: You can write this method without any loops!
        '''
        if X is not None:
            if self.p > 1:
                y_pred = self.make_polynomial_matrix(X, self.p) @ slope + intercept
            else:
                y_pred = (X @ slope) + intercept
        else:
            y_pred = (self.A @ slope) + intercept
        
        return y_pred
            
    def r_squared(self, y_pred):
        '''Computes the R^2 quality of fit statistic

        Parameters:
        -----------
        y_pred: ndarray. shape=(num_data_samps,).
            Dependent variable values predicted by the linear regression model

        Returns:
        -----------
        R2: float.
            The R^2 statistic
        '''
        # Find the mean value of y
        y_mean = np.sum(self.y)/self.y.shape[0]

        # Find the residues between y and y_mean
        mean_res = self.y - y_mean

        # Compute error of model over mean
        E = np.sum(np.square(self.residuals))

        # Compute error of data over mean
        S = np.sum(np.square(mean_res))

        # Compute R^2
        R2 = 1 - E/S

        return R2

    def compute_residuals(self, y_pred):
        '''Determines the residual values from the linear regression model

        Parameters:
        -----------
        y_pred: ndarray. shape=(num_data_samps, 1).
            Data column for model predicted dependent variable values.

        Returns
        -----------
        residuals: ndarray. shape=(num_data_samps, 1)
            Difference between the y values and the ones predicted by the regression model at the 
            data samples
        '''
        res = self.y - y_pred

        return res

    def mean_sse(self, X=None):
        '''Computes the mean sum-of-squares error in the predicted y compared the actual y values.
        See notebook for equation.

        Parameters:
        -----------
        X: ndarray. shape=(anything, num_ind_vars)
            Data to get regression predictions on.
            If None, get predictions based on data used to fit model.

        Returns:
        -----------
        float. Mean sum-of-squares error

        Hint: Make use of self.compute_residuals
        '''

        if X is None:
            y_pred = self.predict(self.slope, self.intercept)
        else:
            y_pred = self.predict(self.slope, self.intercept, X)
        res = self.compute_residuals(y_pred)
        msse = np.mean(np.square(res))
        return msse

    def scatter(self, ind_var, dep_var, title, ind_var_index=0):
        '''Creates a scatter plot with a regression line to visualize the model fit.
        Assumes linear regression has been already run.
        
        Parameters:
        -----------
        ind_var: string. Independent variable name
        dep_var: string. Dependent variable name
        title: string. Title for the plot
        ind_var_index: int. Index of the independent variable in self.slope
            (which regression slope is the right one for the selected independent variable
            being plotted?)
            By default, assuming it is at index 0.

        TODO:
        - Use your scatter() in Analysis to handle the plotting of points. Note that it returns
        the (x, y) coordinates of the points.
        - Sample evenly spaced x values for the regression line between the min and max x data values
        - Use your regression slope, intercept, and x sample points to solve for the y values on the
        regression line.
        - Plot the line on top of the scatterplot.
        - Make sure that your plot has a title (with R^2 value in it)
        '''
        new_title = title + ' with R^2 = ' + str(round(self.R2, 2))
        x, y = super().scatter(ind_var, dep_var, new_title)
        line_x = np.linspace(np.min(x), np.max(x))
        if self.p > 1:
            lineM_x = np.ones([len(line_x), self.p])
            for i in range(self.p):
                lineM_x[:, i] = np.power(line_x, i+1)
            line_y = lineM_x @ self.slope + self.intercept
        else:
            line_y = line_x * self.slope[ind_var_index] + self.intercept
        line = plt.plot(line_x, line_y, c='green')

    def pair_plot(self, data_vars, fig_sz=(12, 12)):
        '''Makes a pair plot with regression lines in each panel.
        There should be a len(data_vars) x len(data_vars) grid of plots, show all variable pairs
        on x and y axes.

        Parameters:
        -----------
        data_vars: Python list of strings. Variable names in self.data to include in the pair plot.
        fig_sz: tuple. len(fig_sz)=2. Width and height of the whole pair plot figure.
            This is useful to change if your pair plot looks enormous or tiny in your notebook!

        TODO:
        - Use your pair_plot() in Analysis to take care of making the grid of scatter plots.
        Note that this method returns the figure and axes array that you will need to superimpose
        the regression lines on each subplot panel.
        - In each subpanel, plot a regression line of the ind and dep variable. Follow the approach
        that you used for self.scatter. Note that here you will need to fit a new regression for
        every ind and dep variable pair.
        - Make sure that each plot has a title (with R^2 value in it)
        '''

        M = len(data_vars)
        fig, ax = super().pair_plot(data_vars, fig_sz)
        for i in range(M):
            for j in range(M):
                x = self.data.select_data(data_vars[j])
                y = self.data.select_data(data_vars[i])
                # Below added for 2c, for 2b please comment out
                if i == j:
                    ax[i, j].clear()
                    ax[i, j].hist(x)
                    if i==0:
                        ax[i, j].set_ylabel(data_vars[i])
                    if i==M-1:
                        ax[i, j].set_xlabel(data_vars[j])
                else:
                # Above added for 2c, for 2b please comment out and unindent the lines below
                    self.linear_regression(data_vars[j], data_vars[i])
                    line_x = np.linspace(np.min(x), np.max(x))
                    line_y = line_x * self.slope[0] + self.intercept
                    ax[i, j].plot(line_x, line_y, c='green')
                    ax[i, j].set_title('R^2 = ' + str(round(self.R2, 2)))

    def make_polynomial_matrix(self, A, p):
        '''Takes an independent variable data column vector `A and transforms it into a matrix appropriate
        for a polynomial regression model of degree `p`.
        
        (Week 2)

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, 1)
            Independent variable data column vector x
        p: int. Degree of polynomial regression model.

        Returns:
        -----------
        ndarray. shape=(num_data_samps, p)
            Independent variable data transformed for polynomial model.
            Example: if p=10, then the model should have terms in your regression model for
            x^1, x^2, ..., x^9, x^10.

        NOTE: There should not be a intercept term ("x^0"), the linear regression solver method
        will take care of that.
        '''

        Ap = np.ones([A.shape[0], p])
        for i in range(p):
            Ap[:, i] = np.power(A.squeeze(), i+1)
        
        return Ap

    def poly_regression(self, ind_var, dep_var, p, method='normal'):
        '''Perform polynomial regression — generalizes self.linear_regression to polynomial curves
        
        (Week 2)
        
        NOTE: For single linear regression only (one independent variable only)

        Parameters:
        -----------
        ind_var: str. Independent variable entered in the single regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. Dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.
        p: int. Degree of polynomial regression model.
            Example: if p=10, then the model should have terms in your regression model for
            x^1, x^2, ..., x^9, x^10
            (The method that you call for the linear regression solver will take care of the intercept)
        method: str. Method used to compute the linear regression. Here are the options:
            'scipy': Use scipy's linregress function.
            'normal': Use normal equations.
            'qr': Use QR factorization (linear algebra section only).

        TODO:
        - This method should mirror the structure of self.linear_regression (compute all the same things)
        - Differences are:
            - You create the independent variable data matrix (self.A) with columns appropriate for
            polynomial regresssion. Do this with self.make_polynomial_matrix
            - You should programatically generate independent variable name strings based on the
            polynomial degree.
                Example: ['X_p1, X_p2, X_p3'] for a cubic polynomial model
            - You set the instance variable for the polynomial regression degree (self.p)
        '''
        # if ind_var in self.data.get_headers():
        #     ind_list = ''
        #     for i in range(p):
        #         ind_list += ind_var + '_p' + str(i+1) + ', '
        #     self.ind_vars = [ind_list]
        # else:
        #     print("Invalid independent variable")
        #     return
        
        # if dep_var in self.data.get_headers():
        #     self.dep_var = dep_var
        # else:
        #     print("Invalid dependent variable")
        #     return

        ind_list = ''
        for i in range(p):
            ind_list += ind_var + '_p' + str(i+1) + ', '
        self.ind_vars = [ind_list]

        self.dep_var = dep_var

        # Find independent variables and dependent variables
        self.A = self.make_polynomial_matrix(self.data.select_data(ind_var), p)
        self.y = self.data.select_data(self.dep_var)

        # Perform linear regression
        if method == 'scipy':
            c = self.linear_regression_scipy(self.A, self.y)
        elif method == 'normal':
            c = self.linear_regression_normal(self.A, self.y)
        elif method == 'qr':
            c = self.linear_regression_qr(self.A, self.y)
        else:
            print('Invalid method')
            return

        self.slope = c[:-1].reshape(len(c)-1,1)
        self.intercept = float(c[-1])

        y_pred = self.predict(self.slope, self.intercept)

        # compute R^2 on the fit and the residuals
        self.residuals = self.compute_residuals(y_pred)
        self.R2 = self.r_squared(y_pred)

        # set the instance variable for the polynomial regression degree
        self.p = p
    
    # Extension 2
    def condition_number(self, A):
        '''Compute matrix condition number of the input matrix

        Parameters:
        -----------
        A: ndarray.
            Matrix whose matrix condition number will be computed.

        Returns
        -----------
        cond_num: float.
            Condition number of the matrix A
        '''
        cond_num = np.linalg.cond(A)
        return cond_num
    
    # Extension 4
    def poly_scatter(self, train_data, test_data, ind_var, dep_var, p):
        '''Creates a scatter plot of both training data and testing data 
        with a group of polynomial regression lines from p=0 to p=p 
        to visualize the model fit.
        
        Parameters:
        -----------
        train_data: Data object. Data for training to find the regression model
        test_data: Data object. Data for testing the regression model
        ind_var: string. Independent variable name
        dep_var: string. Dependent variable name
        p: int. The highest polynomial degree
        '''

        A = train_data
        AT = test_data

        N = A.get_num_samples()
        NT = AT.get_num_samples()

        x = A.select_data(ind_var)
        y = A.select_data(dep_var)
        xT = AT.select_data(ind_var)
        yT = AT.select_data(dep_var)

        line_x = np.linspace(np.min(x)-.1, np.max(x)+.1, num=100)

        Ap = self.make_polynomial_matrix(x, p)
        homo = np.ones([Ap.shape[0],1])
        Ap = np.hstack((homo, Ap))
        ATp = self.make_polynomial_matrix(xT, p)
        homoT = np.ones([ATp.shape[0],1])
        ATp = np.hstack((homoT, ATp))
        lineM_x = self.make_polynomial_matrix(line_x.reshape(100,1), p)
        homoM = np.ones([lineM_x.shape[0],1])
        lineM_x = np.hstack((homoM, lineM_x))
        
        y_mean = np.mean(y)
        mean_res = y - y_mean
        S = np.sum(np.square(mean_res))

        y_meanT = np.mean(yT)
        mean_resT = yT - y_meanT
        ST = np.sum(np.square(mean_resT))

        plt.scatter(x, y, label = 'Training')
        plt.scatter(xT, yT, label = 'Testing')

        for i in np.arange(1,p+2):
            c, res, rnk, s = scipy.linalg.lstsq(Ap[:,:i], y)
            res = np.squeeze(res)

            E = res
            R2 = 1 - E/S

            y_pred = ATp[:,:i] @ c
            resT = yT - y_pred
            ET = np.sum(np.square(resT))
            R2T = 1 - ET/ST

            ind = lineM_x[:,:i] @ c
            plt.plot(line_x, ind, label = "p=" + str(rnk-1) + ": r2 =" + str(round(R2, 2)) + "; r2_test = " + str(round(R2T, 2)))
            
        plt.legend(bbox_to_anchor=(1.01, 1))
        plt.title("Poly regression from p=0 to p=" + str(p))
        plt.xlabel(ind_var)
        plt.ylabel(dep_var)
        plt.show()

