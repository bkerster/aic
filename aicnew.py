# aicnew: python style
from __future__ import division
import numpy as np
from collections import defaultdict

def aicmle(timeSeries, distribution):
    mlevals = {} 
    if distribution == 'pareto':
        mlevals['xmin'] = np.min(timeSeries)
        mlevals['mu'] = 1 - timeSeries.shape[0] / (timeSeries.shape[0] * np.log(mlevals['xmin']) - np.sum(np.log(timeSeries)))
        
    elif distribution == 'lognormal':
        mlevals['mu'] = np.sum(np.log(timeSeries)) / timeSeries.shape[0]
        mlevals['sigma'] = np.sqrt(np.sum( (np.log(timeSeries) - mlevals['mu'])**2) / timeSeries.shape[0])
    
    elif distribution == 'normal':
        mlevals['mu'] = np.mean(timeSeries)
        mlevals['sigma'] = np.sqrt(sum((timeSeries - np.mean(timeSeries))**2) / timeSeries.shape[0])
    
    elif distribution == 'exponential':
        mlevals['lambda'] = 1.0 / np.mean(timeSeries)
        
 return mlevals
 
 
 def aiclike(timeSeries, params, distribution):
    if distribution == 'pareto':
        nloglval = -(timeSeries.shape[0] * np.log(params['mu']) + timeSeries.shape[0] * params['mu'] * np.log(params['xmin']) - (params['xmin']+1) * np.sum(np.log(timeSeries)))
        return nloglval
        
    elif distribution == 'lognormal':
        nloglval = np.sum(np.log(timeSeries * params['sigma'] * np.sqrt(2*np.pi)) + (np.log(timeSeries) - params['mu'])**2 / (2 * params['sigma']**2))
        return nloglval
        
    elif distribution == 'normal':
        nloglval = np.sum(np.log( params['sigma'] * np.sqrt(2*np.pi) ) + (timeSeries - params['mu'])**2 / (2 * params['sigma']**2))
        return nloglval
        
    elif distribution == 'exponential':
        nloglval = np.sum(params['lambda'] * timeSeries - np.log(params['lambda']))
        return nloglval
 
 
def aicpdf(xvals, distribution, params):
    """ Generates the values for the probability distributions """
    if distribution == 'pareto':
        pvals = (params['xmin'] * params['mu'] ** params['xmin']) / (xvals ** (params['xmin'] + 1))
        return pvals
    
    elif distribution == 'lognormal':
        pvals = np.exp(-(np.log(xvals) - params['mu'])**2 / (2 * params['sigma']**2)) / (xvals * params['sigma'] * np.sqrt(2*np.pi))
        return pvals
    
    elif distribution == 'normal':
        pvals = np.exp(-(xvals - params['mu'])**2 / (2 * params['sigma']**2)) / (params['sigma'] * np.sqrt(2*np.pi))
        return pvals
    
    elif distribution == 'exponential':
        pvals = params['lambda'] * np.exp(-params['lambda'] * xvals)
        return pvals 

        
def aic(timeSeries, ssc=0):
    """ Calculates Akaike's Information Criterion for the following distribtions:
  		Guassian/Normal, Lognormal, Exponential, Pareto.
        Gamma is not included at this time.
        
        Paramters:
            timeSeries : numpy array
                1d array of positive values
            ssc : 0 or 1
                flag to force the small sample correction, which will be 
                enabled regardless for less than 40 data points
                
        Output: a diectionary with an entry for each candidate distribution
                  [plots: substructure containing the vectors used to generate plots]
                  [mle: contains the maximum likelihood estimate parameters for each
                        distribution ]
                  [nll: negative log likelihood values for each candidate distribution]
                  [aic: akaike's information criterion for each candidate distribution]
                  [aicdiff: difference scores for the AIC estimates]
                  [weight: AIC weights (1 is likely, 0 is unlikely)]
                  
      This code is based off of aicnew.m written by Theo Rhodes
      """  
    if np.min(timeSeries) <= 0:
        timeSeries = timeSeries + -np.min(timeSeries) + .01
    
    # create histogram to determine plot values
    # note that the original uses hist centers, this uses edges. It may matter
    counts, plotvals_edges = np.histogram(timeSeries, 50)
    plotvals = [np.mean([plotvals_edges[i], plotvals_edges[i+1]]) for i in range(plotvals_edges.shape[0]-1)]
    
    distributions = ['normal', 'lognormal', 'exponential', 'pareto', 'gamma']
    #pdfs = [dict(name=dist) for dist in distributions]
    pdfs = defaultdict(dict)
    aicvals = defaultdict(dict)
    
    # calculate maximum likelihood for core distributions
    # calculate log likelihood value at maximum
    # find k (number of params)
    # generate probability density function using parameters
    pdfs = defaultdict(dict)
    kvals = dict()
    for dist in distributions:
        aicvals[dist]['mle'] = aicmle(timeSeries, dist)
        aicvals[dist]['nll'] = aiclike(timeSeries, aicvals[dist]['mle'], dist)
        kvals[dist] = len(aicvals[dist]['mle'])
        pdfs[dist]['vals'] = aicpdf(plotvals, dist, aicvals[dist]['mle'])
    
    # plot histogram and mle pdf
    # note: only creats the data to make a plot, does not actually generate it
    for dist in distributions:
        scaling = np.sum(counts) / np.sum(pdfs[dist]['vals'])
        aicvals[dist]['plots'] = {}
        aicvals[dist]['plots']['xvals'] = plotvals
        aicvals[dist]['plots']['datay'] = counts
        aicvals[dist]['plots']['aicy'] = pdfs[dist]['vals'] * scaling
        
    # check for small sample correction
    if timeSeries.shape[0] / np.max(kvals.values()) < 40:
        ssc = 1
    
    # calculate akaike information criteria
    for dist in distributions:
        aicvals[dist]['aic'] = 2 * aicvals[dist]['nll'] + 2 * kvals[dist]
        if ssc == 1:
            aicvals[dist]['aic'] = aicvals[dist]['aic'] + 2 * kvals[dist] * (kvals[dist] + 1) / (timeSeries.shape[0] - kvals[dist] -1)
    
    # calculate AIC differences and akaike weights
    aicmin = np.min([aicvals[dist]['aic'] for dist in distributions])
    for dist in distributions:
        aicvals[dist]['aicdiff'] = aicvals[dist]['aic'] - aicmin
    
    aicsum = 0
    for dist in distributions:
        aicsum = aicsum + np.exp(-aicvals[dist]['aicdiff'] / 2)
    
    for dist in distributions:
        aicvals[dist]['weight'] = np.exp(-aicvals[dist]['aicdiff'] / 2) / aicsum
        
    max_weight_val = np.max([aicvals[dist]['weight'] for dist in distributions])
    max_weight = [key for key, value in aicvals.items() if value['weight'] == max_weight_val][0]
    max_weight_params = aicvals[max_weight]['mle']
    
    return aicvals, max_weight, max_weight_params