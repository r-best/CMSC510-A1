##
# Robert Best
# V00742880
# Classes 0 and 8
##

from keras.datasets import mnist
import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import theano.tensor as T
import random
import math
from pprint import pprint

def Arodz(X, Y):
    numberOfFeatures = len(X[0])

    # instantiate an empty PyMC3 model
    basic_model = pm.Model()

    # fill the model with details:
    with basic_model:
        mu_prior_cov = 100*np.eye(numberOfFeatures)
        mu_prior_mu = np.zeros((numberOfFeatures,))
        
        # Priors for w,b (Gaussian priors), centered at 0, with very large std.dev.
        w = pm.MvNormal('estimated_w', mu=mu_prior_mu, cov=mu_prior_cov, shape=numberOfFeatures)
        b  = pm.Normal('estimated_b',0,100)

        # calculate u=w^Tx+b
        ww=pm.Deterministic('my_w_as_mx',T.shape_padright(w,1))
        
        # here w, b are unknown to be estimated from data
        # X is the known data matrix [samples x features]
        u = pm.Deterministic('my_u',T.dot(X,ww) + b)
        # u = pm.Deterministic('my_u',X*w + b);
        
        # P(+1|x)=a(u) #see slides for def. of a(u)
        prob = pm.Deterministic('my_prob',1.0 / (1.0 + T.exp(-1.0*u)))
        
        # class +1 is comes from a probability distribution with probability "prob" for +1, and 1-prob for class 0
        # here Y is the known vector of classes
        # prob is (indirectly coming from the estimate of w,b and the data x)
        Y_obs=pm.Bernoulli('Y_obs',p=prob,observed = Y)
    # done with setting up the model

    # now perform maximum likelihood (actually, maximum a posteriori (MAP), since we have priors) estimation
    # map_estimate1 is a dictionary: "parameter name" -> "it's estimated value"
    map_estimate1 = pm.find_MAP(model=basic_model)
    # pprint(map_estimate1)

    return map_estimate1['estimated_w'], map_estimate1['estimated_b']


def featureSelection_flat(data, targetSize=50):
    """Takes in an array of samples and trims off features
    with low appearance rates until 50 or less remain

    # Arguments
        data: the array of samples, each sample being an array of integers
        targetSize: the target number of features, default 50
    
    # Returns
        The input data reduced to 50 features
    """
    numFeatures = len(data[0])
    numToRemove = numFeatures - targetSize

    # If nothing to remove, we're done
    if numToRemove <= 0:
        return data
    
    # Calculate frequency counts of all features
    featureCounts = np.zeros(numFeatures)
    for sample in data:
        for i, feature in enumerate(sample):
            if feature > 0:
                featureCounts[i] += 1
    
    # Append the least frequently occurring element
    indexesToRemove = []
    while numToRemove > 0:
        
        minIndex = (-1, len(data)+1)
        for i, _ in enumerate(featureCounts):
            if featureCounts[i] < featureCounts[minIndex]:
                minIndex = i
                featureCounts[i] = len(data)+1
        indexesToRemove.append(minIndex)
        numToRemove -= 1
    print(indexesToRemove)
    # Return the dataset without the features at the indexes included in indexesToRemove
    return [[x for i, x in enumerate(sample) if i not in indexesToRemove] for sample in data]


def preprocess(X, Y, C0, C1):
    """Takes in a dataset from keras.datasets.mnist in two arrays, 
    one with samples and the other with the labels at corresponding indices, 
    and applies p-reprocessing rules, including reducing the samples to only those
    labelled with C0 or C1, flattening the 2D samples, and normalizing sample values
    into the [0, 1] range.

    # Arguments
        X: The array of MNIST samples
        Y: The array of MNIST labels
        C0: The label of class 0
        C1: The label of class 1
    
    # Returns
        X: The preprocessed sample set
        Y: The preprocessed label set
    """
    # Filter the datasets down to just the required classes
    X = [_ for i, _ in enumerate(X) if Y[i] == C0 or Y[i] == C1]
    Y = [y for y in Y if y == C0 or y == C1]
    
    # Flatten the 2D representations of the samlpes into 1D arrays
    X = np.reshape(X, (len(X), len(X[0])*len(X[0])))

    # Normalize sample values to be between 0 and 1
    # X = [[x/256 for x in sample] for sample in X]
    
    # Normalize class labels to be 0 and 1
    Y = [0 if y == C0 else 1 for y in Y]

    return X, Y


def main():
    C0 = 0
    C1 = 8

    # Load the train and test sets from MNIST
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Apply preprocessing to the training and test sets
    x_train, y_train = preprocess(x_train, y_train, C0, C1)
    # x_test, y_test = preprocess(x_test, y_test, C0, C1)
    
    # x_train = featureSelection_flat(x_train)
    # print(x_train[0])
    
    print(len(x_train), len(x_train[0]))

    sample_size = len(x_train)
    x_train_sample = x_train[:sample_size]
    y_train_sample = np.array(y_train[:sample_size]).reshape(sample_size, 1)

    Arodz(x_train_sample, y_train_sample)


if __name__ == '__main__':
    np.set_printoptions(linewidth=500)
    main()
