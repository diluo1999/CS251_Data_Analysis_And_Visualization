'''test05_week2.py
Test week 2 methods
CS 251 Data Analysis and Visualization
Spring 2020
Oliver Layton, Caitrin Eaton, Hannah Wolfe
'''
import numpy as np

from data import Data


def test_week2_methods(iris_filename):
    iris_data = Data(iris_filename)
    iris_data_copy = iris_data.get_all_data()
    print(f"Your iris copy has data:\n{iris_data_copy}")

    iris_data_head = iris_data.head()
    print(f'Your iris head should return first 5 rows, which are\n{iris_data.data[:5, :]}\nAnd yours are\n{iris_data_head}')

    iris_data_tail = iris_data.tail()
    print(f'Your iris head should return last 5 rows, which are\n{iris_data.data[-5:, :]}\nAnd yours are\n{iris_data_tail}')

    headers = ['sepal_length', 'petal_length']
    rows = [0,2]
    sample = iris_data.select_data(headers, rows)
    print(f'Your sample is\n{sample}\n\nIt should look like\n[[5.1\t1.4\n4.7\t1.3]]\n')
    empty_rows=[]
    empty_rows_sample = iris_data.select_data(headers, empty_rows)
    print(f'\nIf no rows, Your sample is\n{empty_rows_sample}\n')

if __name__ == '__main__':
    print('---------------------------------------------------------------------------------------')
    print('Begining test 1 (Test all get methods)...')
    print('---------------------------------------------')
    data_file = 'data/iris.csv'
    test_week2_methods(data_file)
    print('---------------------------------------------')
    print('Finished test 1!')
    print('---------------------------------------------------------------------------------------')
