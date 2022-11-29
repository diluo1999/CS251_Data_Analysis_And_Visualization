'''test02_read_data_errors.py
Test Data class error handling
CS 251 Data Analysis and Visualization
Spring 2020
Oliver Layton, Caitrin Eaton, Hannah Wolfe
'''
import numpy as np

from data import Data


def read_data_error(iris_filename):
    iris_data = Data()
    iris_data.read(iris_filename)


if __name__ == '__main__':
    print('---------------------------------------------------------------------------------------')
    print('Begining test 1 (CSV error handling)...')
    print('This should crash, but with your own error message that helps the user identify the problem and what to do to fix it.')
    print('You should identify the reason for this crash and alert the user what it is and how to correct it!')
    print('------------------')
    data_file = 'data/iris_bad.csv'
    read_data_error(data_file)
    print('------------------')
    print('Finished test 1!')
    print('---------------------------------------------------------------------------------------')
