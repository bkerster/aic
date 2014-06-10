# aicmle python style
import numpy as np

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