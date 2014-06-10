#aicpdf python style
import numpy as np
from __future__ import division

def aicpdf(xvals, distribution, params):
    
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