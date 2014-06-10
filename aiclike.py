#aiclike python style
import numpy as np
from __future__ import division

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
        