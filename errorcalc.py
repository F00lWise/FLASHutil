import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.signal import butter, filtfilt, freqz
import pickle
#import deepdish as dd
import h5py

"""
def weighted_avg_and_std(values, weights):
    #formula: https://stats.stackexchange.com/questions/6534/how-do-i-calculate-a-weighted-standard-deviation-in-excel
    weights = np.array(weights)/np.max(weights)
        
    #def weighed_sum()
    if np.any(weights) == 0:
        print('Careful, one or more of your weights are 0!')
    weighted_average = np.average(values, weights=weights)
    weighted_variance = np.average((values-weighted_average)**2, weights=weights)*np.sum(weights)
    weighted_std = np.sqrt(weighted_variance)/ np.sqrt( ((len(weights)-1) / len(weights)) * np.sum(weights) )
    return(weighted_average, weighted_std)
"""
def weighed_mean_and_std(y, yweights, bias_correction = False):
    """Returns the weighed arithmetic mean and std for a series of datapoints y[i],
    each with a certainty of yweights[i]. The Bias correction is only valid if the weights
    are given as frequency weights, corresponding to the number of times the value of y has been
    measured at the respective value.
    Sanity Check:
    >>> np.mean(np.array([2,2,4,5,5,5]))
        3.8333333333333335
    >>> np.std(np.array([2,2,4,5,5,5]))
        1.3437096247164249
    >>> weighed_mean_and_std(np.array([2,2,4,5,5,5]),np.array([1,1,1,1,1,1]))
        (3.8333333333333335, 1.3437096247164249)
    >>> weighed_mean_and_std(np.array([2,4,5]),np.array([2,1,3]))
        (3.8333333333333335, 1.3437096247164249)
        
    Source: https://stats.stackexchange.com/questions/51442/weighted-variance-one-more-time
    
    Further, if the weights represent the inverse variance of the distributions
    from which all points are drawn, the weighed mean becomes the maximum likelyhood estimator 
    of the mean of these probability distributions, given that they in turn are independent and normally
    distributed.
    Sources:
    https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Variance_weights
    https://en.wikipedia.org/wiki/Weighted_least_squares
    """
    weighed_mean = np.sum(y*yweights)/(np.sum(yweights)-int(bias_correction))
    weighed_variance = np.sum(yweights*(y-weighed_mean)**2)/(np.sum(yweights)-int(bias_correction))
    
    return weighed_mean, np.sqrt(weighed_variance)


def pearsonr_ci(x,y,alpha=0.045):
    ''' calculate Pearson correlation along with the confidence interval using scipy and numpy
    Parameters
    ----------
    x, y : iterable object such as a list or np.array
      Input for correlation calculation
    alpha : float
      Significance level. 0.05 by default
    Returns
    -------
    r : float
      Pearson's correlation coefficient
    pval : float
      The corresponding p value
    lo, hi : float
      The lower and upper bound of confidence intervals
    '''

    r, p = scipy.stats.pearsonr(x,y)
    r_z = np.arctanh(r)
    se = 1/np.sqrt(x.size-3)
    z = scipy.stats.norm.ppf(1-alpha/2)
    lo_z, hi_z = r_z-z*se, r_z+z*se
    lo, hi = np.tanh((lo_z, hi_z))
    return r, p, lo, hi

def log10_with_err(X,Xerr):
    logerr = np.abs(Xerr/(X*np.log(10)))
    return np.log10(X), logerr

def ratio_with_error(x,y,xerr,yerr):
    X = x.copy()
    Y = y.copy()
    XERR = xerr.copy()
    YERR = yerr.copy()
    
    invalid_ratio = np.logical_or(np.isnan(X), np.isnan(Y))
    invalid_ratio = np.logical_or(invalid_ratio, Y==0)

    invalid_error = np.logical_or(np.isnan(XERR),np.isnan(YERR))
    invalid_error = np.logical_or(invalid_error, invalid_ratio)
    
    valid = np.logical_not(np.logical_or(invalid_error,invalid_ratio))
    
    ratio = np.zeros(X.shape)*np.nan
    comberr = np.zeros(X.shape)*np.nan
    
    ratio[np.logical_not(invalid_ratio)] = X[np.logical_not(invalid_ratio)]/Y[np.logical_not(invalid_ratio)]
    comberr[valid] = np.abs(X[valid]/Y[[valid]])*np.sqrt( (XERR[valid]/X[valid])**2 + (YERR[valid]/Y[valid])**2 -\
                                    (2*np.corrcoef(X[valid],Y[valid])[0,1]*XERR[valid]*YERR[valid]) \
                                    /(X[valid]*Y[valid]) )# Error Propagation for division, using covariance calculated from correlation coefficient

    return ratio, comberr

def ratio_with_error_correlated_specific(x,y,xerr,yerr, correls):
    """ratio, comberr = ratio_with_error_correlated_specific(x,y,xerr,yerr, correls)"""
    X = x.copy()
    Y = y.copy()
    XERR = xerr.copy()
    YERR = yerr.copy()
    
    invalid_ratio = np.logical_or(np.isnan(X), np.isnan(Y))
    invalid_ratio = np.logical_or(invalid_ratio, Y==0)

    invalid_error = np.logical_or(np.isnan(XERR),np.isnan(YERR))
    invalid_error = np.logical_or(invalid_error, invalid_ratio)
    
    valid = np.logical_not(np.logical_or(invalid_error,invalid_ratio))
    

    
    if type(X) is list or type(X) is np.ndarray and len(X)>1:
        ratio = np.zeros(len(X))*np.nan
        comberr = np.zeros(len(X))*np.nan
        ratio[np.logical_not(invalid_ratio)] = X[np.logical_not(invalid_ratio)]/Y[np.logical_not(invalid_ratio)]
        comberr[valid] = np.abs(X[valid]/Y[valid])*np.sqrt( (XERR[valid]/X[valid])**2 + (YERR[valid]/Y[valid])**2 -\
                                        (2*correls[valid]*XERR[valid]*YERR[valid]) \
                                        /(X[valid]*Y[valid]) )# Error Propagation for division, using covariance calculated from correlation coefficient
    else:
        ratio= X/Y
        comberr = np.abs(X/Y)*np.sqrt( (XERR/X)**2 + (YERR/Y)**2 -\
                                        (2*correls*XERR*YERR) \
                                        /(X*Y) )# Error Propagation for division, using covariance calculated from correlation coefficient

    return ratio, comberr

def ratio_with_error_uncorrelated(x,y,xerr,yerr):
    X = x.copy()
    Y = y.copy()
    XERR = xerr.copy()
    YERR = yerr.copy()
    
    invalid_ratio = np.logical_or(np.isnan(X), np.isnan(Y))
    invalid_ratio = np.logical_or(invalid_ratio, Y==0)

    invalid_error = np.logical_or(np.isnan(XERR),np.isnan(YERR))
    invalid_error = np.logical_or(invalid_error, invalid_ratio)
    
    valid = np.logical_not(np.logical_or(invalid_error,invalid_ratio))
    
    ratio = np.zeros(X.shape)*np.nan
    comberr = np.zeros(X.shape)*np.nan
    
    ratio[np.logical_not(invalid_ratio)] = X[np.logical_not(invalid_ratio)]/Y[np.logical_not(invalid_ratio)]
    comberr[valid] = np.abs(X[valid]/Y[[valid]])*np.sqrt( (XERR[valid]/X[valid])**2 + (YERR[valid]/Y[valid])**2)
    return ratio, comberr

"""
# This seems wrong
def difference_with_error(x,y,xerr,yerr):
    diff = x-y
    comberr = np.sqrt((x**2)*(xerr**2) + \
                      (y**2)*(yerr**2)) -\
                        2*x*y*np.corrcoef(x,y)[0,1]*xerr*yerr# Error Propagation for division, using covariance calculated from correlation coefficient
    return diff, comberr
    
# Better for uncorrelated:
    def difference_with_error(x,y,xerr,yerr):
        diff = x-y
        comberr = np.sqrt((xerr**2) + (yerr**2))
        return diff, comberr
"""
