from __future__ import division
import numpy as np
from scipy.special import gamma

# Gamma function
def Gamma(x, y):
    z = 1-x-y
    if z>0:
        return gamma(z)
    return float('Inf')

# Estimators of Pareto parameters
def EstimateH(samples, n, delta, A_delta):
    d = 24*np.log(2/delta)/n
    p0 = p_func(samples, 0)
    p1 = p_func(samples, 1)
    if p1 <= d:
        print 'Error: p1={} is too small'.format(p1)
        return
    p_list = [p0, p1]
    k_max = 0
    p = p_func(samples, 2)
    while p > d:
        k_max += 1
        p_list.append(p)
        p = p_func(samples, k_max+2)
    for k1 in range(k_max+1):
        k_opt = k1
        for k2 in range(k1+1, k_max+1):
            p1 = p_func(samples, k1)
            p2 = p_func(samples, k2)
            alpha1 = alpha_func(p_list[k1], p_list[k1+1])
            alpha2 = alpha_func(p_list[k2], p_list[k2+1])
            if np.abs(alpha1-alpha2) > A_delta/np.sqrt(n*p_list[k2+1]):
                break
            if k2 == k_max:
                break
    alpha_est = alpha_func(p_list[k_opt], p_list[k_opt+1])
    return 1/alpha_est

def EstimateH_rough(samples):
    T = len(samples)
    # k_opt = np.ceil(np.log(np.log(T))**2)
    k_opt = 0
    p1 = p_func(samples, k_opt)
    p2 = p_func(samples, k_opt+1)
    alpha_est = alpha_func(p1, p2)
    return 1/alpha_est

def EstimateC(samples, h, b):
    T = len(samples)
    return T**(1/(2*b+1))*(1/T)*np.sum([samples[i]>T**(h/(2*b+1)) for i in range(T)])

# Useful functions
def p_func(samples, k):
    T = len(samples)
    return 1/T*np.sum([samples[i]>np.exp(k) for i in range(T)])

def alpha_func(p1, p2):
    return np.log(p1) - np.log(p2)
