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

from utils import utils


def Arodz(x0, x1):
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

    #compare map_estimate1['estimated_mu1'] with true_mu1
    #same for mu_2, cov


def sampleMean(dataset):
    mean = np.zeros(len(dataset[0]))
    for sample in dataset:
        for i, feature in enumerate(sample):
            mean[i] += feature
    return [x/len(dataset) for x in mean]


def test(m0, m1, cov, testX0, testX1):
    """Takes in a test set and the estimated covariance and class0/1 means.
    Calculates the true values from the test set and determines estimate error.

    Arguments:
        m0: Estimated mean of class 0
        m1: Estimated mean of class 1
        cov: Estimated covariance
        testX0: Test samples from class 0
        testX1: Test samples from class 1
    """
    real_m0 = sampleMean(testX0)
    print("Estimated class 0 mean: ", m0)
    print("Class 0 mean of test set: ", real_m0)
    print("Class 0 mean error", sp.spatial.distance.euclidean(m0, real_m0))
    print()

    real_m1 = sampleMean(testX1)
    print("Estimated class 0 mean: ", m1)
    print("Class 0 mean of test set: ", real_m1)
    print("Class 0 mean error", sp.spatial.distance.euclidean(m1, real_m1))
    print()

    real_cov = np.cov(testX0)
    print("Estimated covariance: ", cov)
    print("True test set covariance: ", real_cov)
    print("Covariance error: ")


def main():
    C0 = 0
    C1 = 8

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

    x0_train = [_ for i, _ in enumerate(x_train) if y_train[i] == 0]
    x1_train = [_ for i, _ in enumerate(x_train) if y_train[i] == 1]
    x0_test = [_ for i, _ in enumerate(x_test) if y_test[i] == 0]
    x1_test = [_ for i, _ in enumerate(x_test) if y_test[i] == 1]
    
    sample_size = 1000
    x0_train_sample = random.sample(x0_train, sample_size)
    x1_train_sample = random.sample(x1_train, sample_size)

    m0, m1, cov = Arodz(x0_train_sample, x1_train_sample)

    test(m0, m1, cov, x0_test, x1_test)


if __name__ == '__main__':
    main()
