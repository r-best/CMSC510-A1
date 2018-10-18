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

from utils import utils


def Arodz(x0, x1):
    """Takes in two sample sets, one from each class, and
    returns the MAP estimates of the means and covariance
    """
    numberOfFeatures = len(x0[0])

    # instantiate an empty PyMC3 model
    basic_model = pm.Model()

    # fill the model with details:
    with basic_model:
        # parameters for priors for gaussian means
        mu_prior_cov = 100*np.eye(numberOfFeatures)
        mu_prior_mu = np.zeros((numberOfFeatures,))

        # Priors for gaussian means (Gaussian prior): mu1 ~ N(mu_prior_mu, mu_prior_cov), mu0 ~ N(mu_prior_mu, mu_prior_cov)
        mu1 = pm.MvNormal('estimated_mu1', mu=mu_prior_mu, cov=mu_prior_cov, shape=numberOfFeatures)
        mu0 = pm.MvNormal('estimated_mu0', mu=mu_prior_mu, cov=mu_prior_cov, shape=numberOfFeatures)

        # Prior for gaussian covariance matrix (LKJ prior):
        # see here for details: http://austinrochford.com/posts/2015-09-16-mvn-pymc3-lkj.html
        # and here: http://docs.pymc.io/notebooks/LKJ.html
        sd_dist = pm.HalfCauchy.dist(beta=2.5, shape=numberOfFeatures)
        chol_packed = pm.LKJCholeskyCov('chol_packed',
            n=numberOfFeatures, eta=2, sd_dist=sd_dist)
        chol = pm.expand_packed_triangular(numberOfFeatures, chol_packed)
        cov_mx = pm.Deterministic('estimated_cov', chol.dot(chol.T))

        # observations x1, x0 are supposed to be P(x|y=class1)=N(mu1,cov_both), P(x|y=class0)=N(mu0,cov_both)
        # here is where the Dataset (x1,x0) comes to influence the choice of paramters (mu1,mu0, cov_both)
        # this is done through the "observed = ..." argument; note that above we didn't have that
        x1_obs = pm.MvNormal('x1', mu=mu1,chol=chol, observed = x1)
        x0_obs = pm.MvNormal('x0', mu=mu0,chol=chol, observed = x0)
    # done with setting up the model

    # now perform maximum likelihood (actually, maximum a posteriori (MAP), since we have priors) estimation
    # map_estimate1 is a dictionary: "parameter name" -> "it's estimated value"
    map_estimate1 = pm.find_MAP(model=basic_model)
    # print(map_estimate1)

    return map_estimate1['estimated_mu0'], map_estimate1['estimated_mu1'], map_estimate1['estimated_cov']


def predict(m0, m1, cov, testX):
    """Takes in the estimated covariance and class means
    and uses them to predict labels for the given test set.

    Arguments:
        m0: array-like (1D)
            Estimated mean of class 0
        m1: array-like (1D)
            Estimated mean of class 1
        cov: array-like (2D)
            Estimated covariance
        testX: array-like (2D)
            Array of test set samples to label
    
    Yields:
        The predicted labels for the test set
    """
    m0 = np.atleast_2d(m0)
    m1 = np.atleast_2d(m1)
    cov = np.linalg.inv(cov)

    for item in testX:
        item = np.atleast_2d(item)

        dist0 = np.subtract(item, m0)
        prob0 = np.matmul(np.matmul(dist0, cov), dist0.T)

        dist1 = np.subtract(item, m1)
        prob1 = np.matmul(np.matmul(dist1, cov), dist1.T)

        if prob0 < prob1:
            yield 0
        if prob0 > prob1:
            yield 1


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
    print("Applying feature selection...")
    x_train, x_test = utils.featureSelection(x_train, x_test)

    # Split training set by class
    x0_train = [_ for i, _ in enumerate(x_train) if y_train[i] == 0]
    x1_train = [_ for i, _ in enumerate(x_train) if y_train[i] == 1]

    # Take random sample of each class of training set
    x0_train_sample = random.sample(x0_train, int(len(x0_train)*sampleSize))
    x1_train_sample = random.sample(x1_train, int(len(x1_train)*sampleSize))

    # Use Dr Arodz's code to get MAP estimate
    m0, m1, cov = Arodz(x0_train_sample, x1_train_sample)

    # Predict labels for test set
    labels = predict(m0, m1, cov, x_test)

    # Evaluate label accuracy
    utils.evaluate(labels, y_test)

if __name__ == '__main__':
    np.set_printoptions(linewidth=500, formatter={'float_kind': lambda x: "{0:0.3f}".format(x)})
    main(sys.argv)
