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


def Arodz(x0, x1, numberOfFeatures):
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
    print(map_estimate1)

    #compare map_estimate1['estimated_mu1'] with true_mu1
    #same for mu_2, cov

    # we can also do MCMC sampling from the distribution over the parameters
    # and e.g. get confidence intervals
    # with basic_model:
    #     # instantiate sampler
    #     step = pm.Slice()

    #     # draw 10000 posterior samples
    #     # can take rather long time
    #     trace = pm.sample(10000, step=step, start=start)

    # pm.traceplot(trace)
    # pm.summary(trace)
    # plt.show()





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
    x_train = [[x for row in item for x in row] for item in x_train]
    x_test = [[x for row in item for x in row] for item in x_test]

    N = len(x_train) # Number of samples
    numberOfFeatures = len(x_train[0]) # Number of features in a sample

    x0 = [_ for i, _ in enumerate(x_train) if y_train[i] == C1]
    x1 = [_ for i, _ in enumerate(x_train) if y_train[i] == C2]
    
    sample_size = 1000
    x0_sample = random.sample(x0, sample_size)
    x1_sample = random.sample(x1, sample_size)

    Arodz(x0_sample, x1_sample, numberOfFeatures)


if __name__ == '__main__':
    main()
