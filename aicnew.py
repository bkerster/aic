# aicnew: python style
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def aicnew(timeSeries, plots, ssc):
    
    if np.min(timeSeries) <= 0:
        timeSeries = timeSeries + -np.min(timeSeries) + .01
    
    # create histogram to determine plot values
    # note that the original uses hist centers, this uses edges. It may matter
    counts, plotvals = np.histogram(timeSeries, 50)
    
    distributions = ['normal', 'lognormal', 'exponential', 'pareto', 'gamma']
    #pdfs = [dict(name=dist) for dist in distributions]
    pdfs = defaultdict(dict)
    aicvals = defaultdict(dict)
    
    # calculate maximum likelihood for core distributions
    # calculate log likelihood value at maximum
    # find k (number of params)
    # generate probability density function using parameters
    kvals = defaultdict(dict)
    for dist in distributions:
        aicvals[dist]['mle'] = aicmle(timeSeries, dist)
        aicvals[dist]['nll'] = aiclike(timeSeries,aicvals[dist]['mle']['params'], dist)
        kvals[dist]['vals'] = aicpdf(plotvals, dist, aicvals[dist]['mle']['params'])
    
    