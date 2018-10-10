#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tarodz
Partially adapted from: http://austinrochford.com/posts/2015-09-16-mvn-pymc3-lkj.html
"""

import pymc3 as pm;
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import theano.tensor as T;

### START OF SOME FAKE DATASET GENERATION
np.random.seed(123)

#number of samples in total
N=1000
numberOfFeatures=2;

#true probability of class 1 and 0
p1=.6;
p0=1.0-p1;

#sample number of samples from class 1 n1 ~ Binomial(N,p1)
n1=sp.stats.binom.rvs(N,p1,size=1);
n0=N-n1;


# true means of the gaussians for class 1 and class 0
# 2 features, i.e. 2D guassians
true_mu1 = np.array([1,1]);
true_mu0 = np.array([-1,-1]);

# true covariance matrix, same for both classes
# 2 features, so covariance is 2x2 matrix

true_cov_sqrt = sp.stats.uniform.rvs(0, 2, size=(numberOfFeatures, numberOfFeatures))
true_cov = np.dot(true_cov_sqrt.T, true_cov_sqrt);

# sample feature vectors (2D) from the true gaussians

x1 = sp.stats.multivariate_normal.rvs(true_mu1, true_cov, size=n1)
x0 = sp.stats.multivariate_normal.rvs(true_mu0, true_cov, size=n0)
#replace the above with vectors from the MNIST dataset, x1 for class A, x0 for class B


#### END OF FAKE DATASET GENERATION

##### for MNIST experiment:
#replace the above with two matrices, x0 and x1 [samples x features] from the MNIST dataset, x1 for class A, x0 for class B


#### START OF MODEL BUILDING AND ESTIMATION

# instantiate an empty PyMC3 model
basic_model = pm.Model()

# fill the model with details:
with basic_model:

    # parameters for priors for gaussian means
    mu_prior_cov = 100*np.eye(numberOfFeatures);
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
    x1_obs = pm.MvNormal('x1', mu=mu1,chol=chol, observed = x1);
    x0_obs = pm.MvNormal('x0', mu=mu0,chol=chol, observed = x0);

    
# done with setting up the model
    
    

# now perform maximum likelihood (actually, maximum a posteriori (MAP), since we have priors) estimation
# map_estimate1 is a dictionary: "parameter name" -> "it's estimated value"
map_estimate1 = pm.find_MAP(model=basic_model)

#compare map_estimate1['estimated_mu1'] with true_mu1
#same for mu_2, cov


# we can also do MCMC sampling from the distribution over the parameters
# and e.g. get confidence intervals
#
with basic_model:

    # obtain starting values via MAP
    start = pm.find_MAP()

    # instantiate sampler
    step = pm.Slice()

    # draw 10000 posterior samples
    # can take rather long time
    trace = pm.sample(10000, step=step, start=start)

pm.traceplot(trace)
pm.summary(trace)
plt.show()
