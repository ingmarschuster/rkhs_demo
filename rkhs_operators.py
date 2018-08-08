#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 17:33:27 2018

@author: Ingmar Schuster
"""

#from __future__ import division, print_function, absolute_import

import autograd.numpy as np
import autograd.scipy as sp

import autograd.scipy.stats as stats

from autograd.numpy import exp, log, sqrt
from autograd.scipy.misc import logsumexp

import pylab as pl

import distributions as dist 

class Kernel(object):
    def mean_emb(self, samps):
        return lambda Y: self.gram(samps, Y).sum()/len(samps)
    
    def mean_emb_len(self, samps):
        return self.gram(samps, samps).sum()/len(samps**2)
    
    def sqdist(self, X, Y):
        """Compute the squared RKHS distance between X and Y after mapping into feature space"""
        
    def k(self, X, Y = None):
        """compute the  diagonal of the gram matrix, i.e. the kernel evaluated at each element of X paired with the corresponding element of Y"""
        raise NotImplementedError()
        
    def gram(self, X, Y = None):
        """compute the gram matrix, i.e. the kernel evaluated at every element of X paired with each element of Y"""
        raise NotImplementedError()
    
    def get_params(self):
        # get unconstrained parameters
        assert()
    
    def set_params(self, params):
        # set unconstrained parameters, possibly transform them
        assert()

class FeatMapKernel(Kernel):
    def __init__(self, feat_map):
        self.features = feat_map
        
    def features_mean(self, samps):
        return self.features(samps).mean(0)
    
    def mean_emb_len(self, samps):
        featue_space_mean = self.features_mean(samps)
        return featue_space_mean.dot(featue_space_mean)
    
    def mean_emb(self, samps):
        featue_space_mean = self.features(samps).mean(0)
        return lambda Y: self.features(Y).dot(featue_space_mean)
    
    def gram(self, X, Y = None):
        f_X = self.features(X)
        if Y is None:
            f_Y = f_X
        else:
            f_Y = self.features(Y)
        return f_X.dot(f_Y.T)
    
    def k(self, X, Y = None):
        f_X = self.features(X)
        if Y is None:
            f_Y = f_X
        else:            
            assert(len(X) == len(Y))
            f_Y = self.features(Y)
            
        
        return np.sum(f_X * f_Y, 1)

class LinearKernel(FeatMapKernel):
    def __init__(self):
        FeatMapKernel.__init__(self, lambda x: x)

class GaussianKernel(Kernel):
    def __init__(self, sigma):
        self.set_params(log(exp(sigma) - 1))
    
    def get_params(self):
        return self.params
    
    def set_params(self, params):
        self.params = np.atleast_1d(params).flatten()[0]
        self.__set_standard_dev(log(exp(self.params) + 1))
    
    def __set_standard_dev(self, sd):
        self._const_factor = -0.5 / sd**2
        self._normalization = (sqrt(2*np.pi)*sd)
        self._log_norm = log(self._normalization)
        
    def gram(self, X, Y=None, logsp = False):
        assert(len(np.shape(X))==2)
        
        # if X=Y, use more efficient pdist call which exploits symmetry
        if Y is None:
            Y = X
        assert(len(np.shape(Y))==2)
        assert(np.shape(X)[1]==np.shape(Y)[1])
#       sq_dists = cdist(X, Y, 'sqeuclidean')
        sq_dists = ((np.tile(X,(Y.shape[0], 1)) - np.repeat(Y, X.shape[0], 0))**2).sum(-1).reshape(Y.shape[0], X.shape[0]).T
    
        if not logsp:
            return exp(self._const_factor* sq_dists)/self._normalization
        else:
            return self._const_factor* sq_dists - self._log_norm
        
    def k(self, X, Y = None, logsp = False):
        if Y is None:
            Y = X
            
        assert(X.shape == Y.shape)
        sq_dists = np.sum((X - Y)**2, 1)
        if not logsp:
            return exp(self._const_factor* sq_dists)/self._normalization
        else:
            return self._const_factor* sq_dists - self._log_norm

class StudentKernel(Kernel):
    def __init__(self, s2, df):
        self.set_params(log(exp(np.array([s2,df])) - 1))
        
    
    def get_params(self):
        return self.params
    
    def set_params(self, params):
        self.params = params
        self.dens = dist.mvt(0, log(exp(params[0]) + 1), log(exp(params[1]) + 1))
    
    def gram(self, X,Y=None, logsp = False):
        if Y is None:
            Y = X
        assert(len(np.shape(Y))==2)
        assert(np.shape(X)[1]==np.shape(Y)[1])
#       sq_dists = cdist(X, Y, 'sqeuclidean')
        sq_dists = ((np.tile(X,(Y.shape[0], 1)) - np.repeat(Y, X.shape[0], 0))**2).sum(-1).reshape(Y.shape[0], X.shape[0]).T
        dists = np.sqrt(sq_dists)
        rval = self.dens.logpdf(dists.flatten()).reshape(dists.shape)
        if not logsp:
            return rval
        else:
            return exp()
    
    def k(self, X, Y = None):
        if Y is None:
            Y = X
            
        assert(X.shape == Y.shape)
        dists = sqrt(np.sum((X - Y)**2, 1).squeeze())
        return exp(self.dens.logpdf(dists.flatten())).reshape(dists.shape)

def approximate_density(test_points, samples, fact, kernel, logspace = False):
    if not logspace:
#        assert()
        return np.squeeze(kernel.gram(np.atleast_2d(test_points), samples)@fact)
    else:
        return logdotexp(kernel.gram(np.atleast_2d(test_points), samples, logsp = True), fact.squeeze())

def density_norm(prefactors, logsp = False, norm_axis = None):
    if not logsp:
        return prefactors / prefactors.sum(norm_axis)
    else:
        return prefactors - logsumexp(prefactors, norm_axis)

def logdotexp(A, B):
    assert(A.shape[-1] == B.shape[0])
    max_A = A.max()
    max_B = B.max()
    return log(np.dot(np.exp(A - max_A), np.exp(B - max_B))) + max_A + max_B

class RKHSDensityEstimator(object):
    def __init__(self, samps, kern, regul):
        self.samps = samps
        self.kern = kern
        self.G = kern.gram(samps)
        self.G_sc_i = np.linalg.inv(self.G / samps.shape[0] + np.eye(self.G.shape[0]) * regul)
        self.G_i = np.linalg.inv(self.G + np.eye(self.G.shape[0]) * regul)
        self.kme_fact = density_norm(np.ones(samps.shape[0])/samps.shape[0])
        if True:
            self.rkhs_dens_fact = density_norm((self.G_i).mean(1))
        elif False:
            self.rkhs_dens_fact = density_norm(self.G_sc_i.mean(1))
        else:
            pass
#        self.rkhs_dens_fact = density_norm(np.ones(self.G.shape[0])/self.G.shape[0]**2)
    
    def eval_kde(self, at_points, logsp = False):
        #evaluate kernel density estimator, which is the same as kernel mean embedding
        return self.eval_kme(at_points, logsp)
    
    def eval_kme(self, at_points, logsp = False):
        return approximate_density(at_points, self.samps, self.kme_fact, self.kern)
    
    def eval_rkhs_density_approx(self, at_points, logsp = False):
        return approximate_density(at_points, self.samps, self.rkhs_dens_fact, self.kern)

class RKHSOperator(object):
    def __init__(self, inp, outp, inp_kern, outp_kern, logsp = False):
        self.inp = inp
        self.logsp = logsp
        self.outp = outp
        self.inp_kern = inp_kern
        self.outp_kern = outp_kern
        
    def lhood_input_output_pairs(self, inp, outp):
        assert(outp.shape == inp.shape)
        #making this more efficient would only compute the diagonal
        #which can only save some additions/multiplications
        return np.diag(self.lhood(inp, outp, self.logsp))
        
    def lhood(self, inp, outp, logsp = False):
        if not self.logsp:
            fact = self.matr @ (self.inp_kern.gram(self.inp, np.atleast_2d(inp)))
            
        else:
            fact = logdotexp(self.logmatr, self.inp_kern.gram(self.inp, np.atleast_2d(inp), self.logsp))
        rval = approximate_density(outp, self.outp, density_norm(fact, self.logsp, 0), self.outp_kern, self.logsp)
        if self.logsp == logsp:
            return rval
        elif logsp:
            return log(rval)
        else:
            return exp(rval)
    

class ConditionMeanEmbedding(RKHSOperator):
    def __init__(self, inp, outp, inp_kern, outp_kern, inp_regul = 0.00001):
        RKHSOperator.__init__(self, inp, outp, inp_kern, outp_kern)
        G_inp = inp_kern.gram(inp)
        self.matr = np.linalg.inv(G_inp+inp_regul * np.eye(G_inp.shape[0]))

class ConditionDensityOperator(RKHSOperator):
    def __init__(self, inp, outp, inp_kern, outp_kern, inp_regul = 0.00001, outp_regul = 0.00001):
        RKHSOperator.__init__(self, inp, outp, inp_kern, outp_kern)
        G_inp = inp_kern.gram(inp)
        G_inp_inv = np.linalg.inv(G_inp + inp_regul * np.eye(G_inp.shape[0]))
        G_outp = outp_kern.gram(outp)
        G_outp_inv = np.linalg.inv(G_outp + outp_regul * np.eye(G_outp.shape[0]))
        self.matr = G_outp_inv.dot(G_inp_inv)

def test_rkhsoperator_logsp():
    a = np.arange(5).astype(np.float).reshape((5, 1))
    i = np.arange(3)
    o = RKHSOperator(a, a**2, inp_kern=GaussianKernel(1), outp_kern=GaussianKernel(1))
    o.matr = np.arange(1., 26.).reshape((5, 5))
    o.logmatr = log(o.matr)
    rs = np.random.RandomState(None)
    
    b = np.reshape(4 + rs.randn(3), (3,1))
    assert(np.allclose(o.lhood(2, b, False),
                          exp(o.lhood(2, b, True))))
    assert(np.allclose(o.lhood(np.arange(2).reshape((2, 1)) * 4 - 2, b, False),
                          exp(o.lhood(np.arange(2).reshape((2, 1)) * 4 - 2, b, True))))
    


def test_rkhs_dens_and_operators(D = 1, nsamps = 200):
    targ = dist.mixt(D, [dist.mvnorm(3*np.ones(D), np.eye(D)*0.7**2), dist.mvnorm(7*np.ones(D), np.eye(D)*1.5**2)], [0.5, 0.5])
    out_samps = targ.rvs(nsamps)
    
    gk_x = GaussianKernel(0.7)
    de = RKHSDensityEstimator(out_samps, gk_x, 0.1)
    
    x = np.linspace(-1,12,200)
    
    pl.figure()
    pl.plot(x, exp(targ.logpdf(x)), 'k-', label='truth')
    pl.plot(x, de.eval_rkhs_density_approx(x[:,None]), 'b--', label='Density estimate')
    pl.plot(x, de.eval_kme(x[:,None]), 'r:', label='KDE/Kernel mean embedding')
    pl.legend(loc='best')
    pl.savefig('Density_estimation_(preimage_of_KDE).pdf')
    
    
    inp_samps = (out_samps-5)**2 + np.random.randn(*out_samps.shape)
    gk_y = GaussianKernel(1)
    
    cme = ConditionMeanEmbedding(inp_samps, out_samps, gk_y, gk_x, 5)
    cdo = ConditionDensityOperator(inp_samps, out_samps, gk_y, gk_x, 5, 5)
    
    
    
    
    (fig, ax) = pl.subplots(3, 1, True, False, figsize=(10,10))
    ax[2].scatter(out_samps, inp_samps, alpha=0.3)
    ax[2].axhline(0, 0, 8, color='r', linestyle='--')
    ax[2].axhline(5, 0, 8, color='r', linestyle='--')
    ax[2].set_title("Input:  y, output: x,  %d pairs"%nsamps)
    ax[2].set_yticks((0, 5))
    d = cdo.lhood(np.array([[0.], [5.]]), x[:, None]).T
    e = cme.lhood(np.array([[0.], [5.]]), x[:, None]).T
    assert(d.shape[0] == 2)
    assert(np.allclose(d[0], cdo.lhood(0, x[:, None])))
    assert(np.allclose(d[1], cdo.lhood(5, x[:, None])))
#    assert()
    ax[1].plot(x, d[1], '-', label='cond. density')
    ax[1].plot(x, e[1], '--', label='cond. mean emb.')
    ax[1].set_title("p(x|y=5)")
    ax[0].plot(x, d[0], '-', label='cond. density')
    ax[0].plot(x, e[0], '--', label='cond. mean emb.')
    ax[0].set_title("p(x|y=0)")
    ax[0].legend(loc='best')
    fig.show()
    fig.savefig("conditional_density_operator.pdf")
