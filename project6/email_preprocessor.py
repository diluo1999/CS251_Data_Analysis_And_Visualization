'''email_preprocessor.py
Preprocess Enron email dataset into features for use in supervised learning algorithms
Di Luo
CS 251 Data Analysis Visualization, Spring 2020
'''
import re
import os
import numpy as np
import math


def tokenize_words(text):
    '''Transforms an email into a list of words.

    Parameters:
    -----------
    text: str. Sentence of text.

    Returns:
    -----------
    Python list of str. Words in the sentence `text`.

    This method is pre-filled for you (shouldn't require modification).
    '''
    # Define words as lowercase text with at least one alphabetic letter
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pattern.findall(text.lower())


def count_words(email_path='data/enron'):
    '''Determine the count of each word in the entire dataset (across all emails)

    Parameters:
    -----------
    email_path: str. Relative path to the email dataset base folder.

    Returns:
    -----------
    word_freq: Python dictionary. Maps words (keys) to their counts (values) across the dataset.
    num_emails: int. Total number of emails in the dataset.

    TODO:
    - Descend into the dataset base directory -> class folders -> individual emails.
    - Read each email file as a string.
    - Use `tokenize_words` to chunk it into a list of words.
    - Update the counts of each words in the dictionary.

    Hints:
    - Check out Python functions in the os and os.path modules for walking the directory structure.
    - When reading in email files, you might experience errors due to reading funky characters
    (spam can contain weird things!). On this dataset, this can be fixed by telling Python to assume
    each file is encoded using 'latin-1': encoding='latin-1'
    '''
    directory = []

    for (root,dirs,files) in os.walk(email_path,topdown=True):
        for x in files:
            if x.endswith(".txt"):
                directory.append(os.path.join(root, x))

    freq = {}

    for paths in directory:
        file = open(paths, encoding = "latin-1").read()
        words = tokenize_words(file)
        for items in words:
            if freq.get(items) is None:
                freq[items] = 1
            else:
                freq[items]+= 1 

    n_file = len(directory)           

    return freq, n_file


def find_top_words(word_freq, num_features=200):
    '''Given the dictionary of the words that appear in the dataset and their respective counts,
    compile a list of the top `num_features` words and their respective counts.

    Parameters:
    -----------
    word_freq: Python dictionary. Maps words (keys) to their counts (values) across the dataset.
    num_features: int. Number of top words to select.

    Returns:
    -----------
    top_words: Python list. Top `num_features` words in high-to-low count order.
    counts: Python list. Counts of the `num_features` words in high-to-low count order.
    '''
    top_words = []
    counts = []
    sort_orders = sorted(word_freq, key=word_freq.get, reverse=True)
    for i in sort_orders:
        top_words.append(i)
        counts.append(word_freq[i])

    return top_words[:num_features],counts


def make_feature_vectors(top_words, num_emails, email_path='data/enron'):
    '''Count the occurance of the top W (`num_features`) words in each individual email, turn into
    a feature vector of counts.

    Parameters:
    -----------
    top_words: Python list. Top `num_features` words in high-to-low count order.
    num_emails: int. Total number of emails in the dataset.
    email_path: str. Relative path to the email dataset base folder.

    Returns:
    -----------
    feats. ndarray. shape=(num_emails, num_features).
        Vector of word counts from the `top_words` list for each email.
    y. ndarray of nonnegative ints. shape=(num_emails,).
        Class index for each email (spam/ham)

    TODO:
    - Descend into the dataset base directory -> class folders -> individual emails.
    - Read each email file as a string.
    - Use `tokenize_words` to chunk it into a list of words.
    - Count the occurance of each word, ONLY THOSE THAT APPEAR IN `top_words`.

    HINTS:
    - Start with your code in `count_words` and modify as needed.
    '''
    y = np.zeros((num_emails))
    feats = np.zeros((num_emails, len(top_words)))
    idx = 0

    for (root,dirs,files) in os.walk(email_path,topdown=True):
        for x in files:
            if x.endswith(".txt"):
                fn = os.path.join(root, x)
                file = open(fn, encoding = "latin-1").read()
                words = tokenize_words(file)
                if 'spam' in fn:
                    y[idx] = 0
                else:
                    y[idx] = 1
                for j in range(len(top_words)):
                    feats[idx, j] = words.count(top_words[j])
                idx += 1

    return feats, y


def make_train_test_sets(features, y, test_prop=0.2, shuffle=True):
    '''Divide up the dataset `features` into subsets ("splits") for training and testing. The size
    of each split is determined by `test_prop`.

    Parameters:
    -----------
    features. ndarray. shape=(num_emails, num_features).
        Vector of word counts from the `top_words` list for each email.
    y. ndarray of nonnegative ints. shape=(num_emails,).
        Class index for each email (spam/ham)
    test_prop: float. Value between 0 and 1. What proportion of the dataset samples should we use
        for the test set? e.g. 0.2 means 20% of samples are used for the test set, the remaining
        80% are used in training.
    shuffle: boolean. Should we shuffle the data before splitting it into train/test sets?

    Returns:
    -----------
    x_train: ndarray. shape=(num_train_samps, num_features).
        Training dataset
    y_train: ndarray. shape=(num_train_samps,).
        Class values for the training set
    inds_train: ndarray. shape=(num_train_samps,).
        The index of each training set email in the original unshuffled dataset.
        For example: if we have originally N=5 emails in the dataset, their indices are
        [0, 1, 2, 3, 4]. Then we shuffle the data. The indices are now [4, 0, 3, 2, 1]
        let's say we put the 1st 3 samples in the training set and the remaining
        ones in the test set. inds_train = [4, 0, 3] and inds_test = [2, 1].
    x_test: ndarray. shape=(num_test_samps, num_features).
        Test dataset
    y_test:ndarray. shape=(num_test_samps,).
        Class values for the test set
    inds_test: ndarray. shape=(num_test_samps,).
        The index of each test set email in the original unshuffled dataset.
        For example: if we have originally N=5 emails in the dataset, their indices are
        [0, 1, 2, 3, 4]. Then we shuffle the data. The indices are now [4, 0, 3, 2, 1]
        let's say we put the 1st 3 samples in the training set and the remaining
        ones in the test set. inds_train = [4, 0, 3] and inds_test = [2, 1].

    HINTS:
    - If you're shuffling, work with indices rather than actual values.
    '''
    N, W = features.shape 
    indices = np.arange(N)
    if shuffle:
        np.random.shuffle(indices)
    train_prop = math.ceil((1- test_prop) * N)
    inds_train = indices[:train_prop]
    x_train = features[inds_train]
    y_train = y[inds_train]

    inds_test = indices[train_prop:]
    x_test = features[inds_test]
    y_test = y[inds_test]

    return x_train, y_train, inds_train, x_test, y_test, inds_test

def retrieve_emails(inds, email_path='data/enron'):
    '''Obtain the text of emails at the indices `inds` in the dataset.

    Parameters:
    -----------
    inds: ndarray of nonnegative ints. shape=(num_inds,).
        The number of ints is user-selected and indices are counted from 0 to num_emails-1
        (counting does NOT reset when switching to emails of another class).
    email_path: str. Relative path to the email dataset base folder.

    Returns:
    -----------
    Python list of str. len = num_inds = len(inds).
        Strings of entire raw emails at the indices in `inds`
    '''
    texts = [] 
    class_fp = os.listdir(email_path)
    class_fp = [filename for filename in class_fp if os.path.isdir(os.path.join(email_path, filename))]
    for i in range(len(class_fp)):
        class_dir = os.path.join(email_path, class_fp[i])
        for email_f in os.listdir(class_dir):
            cur_f = os.path.join(class_dir, email_f)
            if i in inds:
                with open(cur_f, 'r', encoding='latin-1') as fp:
                    email = fp.readlines()
                    email = ''.join(email)
                texts.append(email)
            i += 1
    return texts 
