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

# y_train shape should be samples x 1 (COLUMN MATRIX!!!!!)
def Arodz(X, Y, numberOfFeatures):
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
        # here w, b are unknown to be estimated from data
        # X is the known data matrix [samples x features]
        u = w*X + b
        # P(+1|x)=a(u) #see slides for def. of a(u)
        prob = 1.0 / (1.0 + T.exp(-1.0*u))
        
        # class +1 is comes from a probability distribution with probability "prob" for +1, and 1-prob for class 0
        # here Y is the known vector of classes
        # prob is (indirectly coming from the estimate of w,b and the data x)
        Y_obs=pm.Bernoulli('Y_obs', p=prob, observed=Y)
    # done with setting up the model

    # now perform maximum likelihood (actually, maximum a posteriori (MAP), since we have priors) estimation
    # map_estimate1 is a dictionary: "parameter name" -> "it's estimated value"
    map_estimate1 = pm.find_MAP(model=basic_model)
    pprint(map_estimate1)

    # we can also do MCMC sampling from the distribution over the parameters
    # and e.g. get confidence intervals
    # with basic_model:
    #     # obtain starting values via MAP
    #     start = pm.find_MAP()

    #     # instantiate sampler
    #     step = pm.Slice()

    #     # draw 10000 posterior samples
    #     # can take rather long time
    #     # trace = pm.sample()
    #     trace = pm.sample(10000, step=step, start=start)

    # pm.traceplot(trace)
    # pm.summary(trace)
    # plt.show()


def featureSelection_flat(data, targetSize=50):
    """Takes in an array of samples and trims off features
    with low appearance rates until <=50 remain

    # Arguments
        data: the array of samples, each sample being an array of integers
        targetSize: the target number of features, default 50
    
    # Returns
        The input data reduced to <=50 features
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


C1 = 0
C2 = 8

def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Filter the datasets down to just the required classes
    x_train = [_ for i, _ in enumerate(x_train) if y_train[i] == C1 or y_train[i] == C2]
    y_train = [x for x in y_train if x == C1 or x == C2]
    x_test = [_ for i, _ in enumerate(x_test) if y_test[i] == C1 or y_test[i] == C2]
    y_test = [x for x in y_test if x == C1 or x == C2]

    # Flatten the 2D representations of the samlpes into 1D arrays
    x_train = np.reshape(x_train, (len(x_train), len(x_train[0])*len(x_train[0])))
    x_test = np.reshape(x_test, (len(x_test), len(x_test[0])*len(x_test[0])))


    # Normalize values to be between 0 and 1
    # x_train = [[x/256 for x in sample] for sample in x_train]
    # x_test = [[x/256 for x in sample] for sample in x_test]
    # print(x_train[0])

    # Normalize class labels to be 0 and 1 (only need to change the 8s to 1s)
    y_train = [0 if x == 0 else 1 for x in y_train]
    
    # x_train = featureSelection_flat(x_train)
    # print(x_train[0])

    N = len(x_train) # Number of samples
    numberOfFeatures = len(x_train[0]) # Number of features in a sample
    
    print(N, len(y_train), numberOfFeatures)

    sample_size = len(x_train)
    x_train_sample = x_train[:sample_size]
    y_train_sample = np.array(y_train[:sample_size]).reshape(sample_size, 1)
    print(y_train_sample)

    Arodz(x_train_sample, y_train_sample, numberOfFeatures)


if __name__ == '__main__':
    np.set_printoptions(linewidth=500)
    main()
