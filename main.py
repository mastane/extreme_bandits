from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from run import *
from multiprocessing import Pool
from joblib import Parallel, delayed
import time

## Problem parameters
n = int(1e4)  # time horizon of one experiment
num_iter = 100  # int(1e3)  # number of experiments
K = 3
b = 1
A = 2e-3  # constant used to compute N
N = int(A*np.log(n)**(2*(2*b+1)/b))
delta = np.exp(-np.log(n)**2)/(2*K*n)
A_delta = 1  # constant used to estimate 1/alpha

## Second order Pareto distributions on the arms
# alpha, C: Pareto parameters
alphas = [5, 1.5, 2]
Cs = [1, 1, 1]
scales = [Cs[k]**(1/alphas[k]) for k in range(K)]
# 0 <= pareto_weight <= 1: the distribution is a mixture between an exact Pareto,
# weighted by pareto_weight, and a Dirac in O, weighted by 1-pareto_weight.
pareto_weights = [1, 1, 1]
alpha_min = min(alphas)
best_arm = np.argmin(alphas)  # because smallest alpha and same C and pareto_weight

## Robust UCB parameters
u = 1  # threshold giving MAB formulation
eps = 0.4  # (alpha_min-1)/2  # eps such that 1+eps < alpha
def moment_max():
    # when same C on all arms
    return alpha_min/(alpha_min-1-eps)*Cs[0]**((1+eps)/alpha_min)
v = moment_max()  # moment(1+eps) = alpha/(alpha-1-eps)*C^((1+eps)/alpha) <= v

regret_EH = np.zeros(n)
regret_RUCB = np.zeros(n)
regret_rand = np.zeros(n)  # strategy pulling random arm at each round

# parallelizing
pool = Pool(processes=4)
args = [n, K, N, alphas, scales, pareto_weights, b, A, A_delta, u, eps, v, best_arm, num_iter]
regrets = pool.map(f_run, [[i]+args for i in range(num_iter)])
for i in range(num_iter):
    for t in range(n):
        regret_EH[t] += regrets[i][0][t]
        regret_RUCB[t] += regrets[i][1][t]
        regret_rand[t] += regrets[i][2][t]

# divide by number of experiments
regret_EH = regret_EH/num_iter
regret_RUCB = regret_RUCB/num_iter
regret_rand = regret_rand/num_iter

## Plot expected extreme regret
fig, ax = plt.subplots()
ax.plot(np.arange(1, n+1), regret_EH, '-', label='ExtremeHunter')
ax.plot(np.arange(1, n+1), regret_RUCB, '--', label='Robust UCB')
ax.plot(np.arange(1, n+1), regret_rand, ':', label='random')

# Now add the legend with some customizations.
legend = ax.legend(loc='upper center', shadow=True)

# The frame is matplotlib.patches.Rectangle instance surrounding the legend.
frame = legend.get_frame()
frame.set_facecolor('0.90')

# Set the fontsize
for label in legend.get_texts():
    label.set_fontsize('large')

for label in legend.get_lines():
    label.set_linewidth(1.5)  # the legend line width
plt.xlabel('time')
plt.ylabel('extreme regret')
plt.savefig('EHvsRUCBvsRand_iter{}.png'.format(num_iter), format='png')
