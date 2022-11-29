'''naive_bayes_multinomial.py
Naive Bayes classifier with Multinomial likelihood for discrete features
Di Luo
CS 251 Data Analysis Visualization, Spring 2020
'''
import numpy as np


class NaiveBayes:
    '''Naive Bayes classifier using Multinomial likeilihoods (discrete data belonging to any
     number of classes)'''
    def __init__(self, num_classes):
        '''Naive Bayes constructor sets the number of classes (int: num_classes) this classifier
        will be trained to detect. All other fields initialized to None.'''
        
        self.num_classes = num_classes

        # class_priors: ndarray. shape=(num_classes,).
        #   Probability that a training example belongs to each of the classes
        #   For spam filter: prob training example is spam or ham
        self.class_priors = None
        
        # class_likelihoods: ndarray. shape=(num_classes, num_features).
        #   Probability that each word appears within class c
        self.class_likelihoods = None

    def train(self, data, y):
        '''Train the Naive Bayes classifier so that it records the "statistics" of the training set:
        class priors (i.e. how likely an email is in the training set to be spam or ham?) and the
        class likelihoods (the probability of a word appearing in each class — spam or ham)

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.
        y: ndarray. shape=(num_samps,). Corresponding class of each data sample.

        TODO:
        - Compute the instance variables self.class_priors and self.class_likelihoods needed for
        Bayes Rule. See equations in notebook.
        '''
        self.class_priors = np.zeros((self.num_classes)) 
        self.class_likelihoods = np.zeros((self.num_classes,len(data[0])))
        for i in range (self.num_classes):
            nc = np.count_nonzero(y == i)
            self.class_priors[i] = nc/len(y)
            for j in range (len(data[0])):
                ind = np.where(y == i)            
                ncw = np.sum(data[:,j][ind])
                total = np.sum(data[ind,:])
                self.class_likelihoods[i,j] = (ncw+1)/(total+len(data[0]))

    def predict(self, data):
        '''Combine the class likelihoods and priors to compute the posterior distribution. The
        predicted class for a test sample from `data` is the class that yields the highest posterior
        probability.

        Parameters:
        -----------
        data: ndarray. shape=(num_test_samps, num_features). Data to predict the class of
            Need not be the data used to train the network

        Returns:
        -----------
        ndarray of nonnegative ints. shape=(num_samps,). Predicted class of each test data sample.

        TODO:
        - Process test samples one-by-one.
        - Look up the likelihood (from training) ONLY AT the words that appear > 0 times in the
        current test sample.
        - Take the log and sum these likelihoods together.
        - Solve for posterior for each test sample i (see notebook for equation).
        - Predict the class of each test sample according to the class that produces the largest
        posterior probability.
        '''
        results = np.zeros(N)
         N, M = data.shape
        for i in range(N):
            features = data[i] 
            idx = np.nonzero(features)
            prob = np.zeros(self.num_classes)
            for c in range(self.num_classes):
                prob[c] = np.log(self.class_priors[c]) + np.sum(np.log(features[idx] * (self.class_likelihoods[c])[idx]))
            results[i] = np.argmax(prob)
        return results

    def accuracy(self, y, y_pred):
        '''Computes accuracy based on percent correct: Proportion of predicted class labels `y_pred`
        that match the true values `y`.

        Parameters:
        -----------
        y: ndarray. shape=(num_data_sams,)
            Ground-truth, known class labels for each data sample
        y_pred: ndarray. shape=(num_data_sams,)
            Predicted class labels by the model for each data sample

        Returns:
        -----------
        float. Between 0 and 1. Proportion correct classification.

        NOTE: Can be done without any loops
        '''
        return (y.shape[0]-np.count_nonzero(y-y_pred))/y.shape[0]

    def confusion_matrix(self, y, y_pred):
        '''Create a confusion matrix based on the ground truth class labels (`y`) and those predicted
        by the classifier (`y_pred`).

        Parameters:
        -----------
        y: ndarray. shape=(num_data_samps,)
            Ground-truth, known class labels for each data sample
        y_pred: ndarray. shape=(num_data_samps,)
            Predicted class labels by the model for each data sample

        Returns:
        -----------
        ndarray. shape=(num_classes, num_classes).
            Confusion matrix
        '''
        confusion_matrix = np.zeros([self.num_classes, self.num_classes])
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                confusion_matrix[i, j] = np.count_nonzero(np.logical_and(y == i, y_pred == j))
        return confusion_matrix
