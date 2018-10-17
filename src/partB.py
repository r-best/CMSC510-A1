"""
Robert Best
V00742880 - Classes 0 and 8
"""

from keras.datasets import mnist
import sys
import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import theano.tensor as T
import random
import math
from pprint import pprint

from utils import utils


def Arodz(X, Y):
    numberOfFeatures = len(X[0])
    Y = np.reshape(Y, (len(Y), 1))

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


def test(w, b, testX, testY):
    """Uses the given estimated w and b to label the elements
    of the given test set and checks to see how accurate the results are

    Arguments:
        w: Array of estimated feature weights
        b: Estimated b value
        testX: Array of test set samples
        testY: Array of gold standard test set labels
    """
    true0 = 0
    false0 = 0
    true1 = 0
    false1 = 0
    for i, item in enumerate(testX):
        u = T.dot(item,w) + b
        prob = 1.0 / (1.0 + T.exp(-1.0*u))
        label = math.floor(prob.eval())

        if label == 0:
            if testY[i] == 0:
                true0 += 1
            else:
                false0 += 1
        elif label == 1:
            if testY[i] == 1:
                true1 += 1
            else:
                false1 += 1

    print("----------------------------------------")
    print("|                         Actual       |")
    print("|          ----------------------------|")
    print("|          |     |    0     |     1    |")
    print("|          |-----|---------------------|")
    print("|          |  0  |   {}    |    {}   |".format(true0, false0))
    print("|Predicted |     |          |          |")
    print("|          |  1  |   {}    |    {}   |".format(true1, false1))
    print("----------------------------------------")
    
    print("Class 0 Precision: {:.3f}".format(true0 / (true0 + false0)))
    print("Class 0 Recall: {:.3f}".format(true0 / (true0 + false1)))
    print("Class 1 Precision: {:.3f}".format(true1 / (true1 + false1)))
    print("Class 1 Recall: {:.3f}".format(true1 / (true1 + false0)))
    print("Accuracy: {}/{} = {:.3f}%".format(true0+true1, len(testY), (true0+true1)/len(testY)*100))


def main(argv):
    C0 = 0
    C1 = 8

    # Read args from command line
    sampleSize = utils.parseArgs(argv)

    # Load the train and test sets from MNIST
    print("Loading datasets from MNIST...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Apply preprocessing to the training and test sets
    print("Preprocessing training set...")
    x_train, y_train = utils.preprocess(x_train, y_train, C0, C1)
    print("Preprocessing testing set...")
    x_test, y_test = utils.preprocess(x_test, y_test, C0, C1)
    
    # Apply feature selection to training set
    # print("Applying feature selection...")
    # x_train, x_test = utils.featureSelection(x_train, x_test)

    # Sample training set
    sampleIndicies = random.sample(range(len(x_train)), int(len(x_train)*sampleSize))
    x_train_sample = [_ for i, _ in enumerate(x_train) if i in sampleIndicies]
    y_train_sample = [_ for i, _ in enumerate(y_train) if i in sampleIndicies]


    # Obtain MAP estimates
    print("Running Dr Arodz's code to obtain MAP estimates of w and b")
    w, b = Arodz(x_train_sample, y_train_sample)

    # Evaluate model accuracy
    print("Testing model...")
    test(w, b, x_test, y_test)


if __name__ == '__main__':
    np.set_printoptions(linewidth=500)
    main(sys.argv)
