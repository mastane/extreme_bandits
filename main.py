from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from arm import *
from sample import *

## Problem parameters
n = int(1e4)  # time horizon of one experiment
num_iter = 1  # int(1e3)  # number of experiments
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
# 0 <= pareto_weight <= 1: the distribution is a mixture between an exact Pareto, weighted by pareto_weight,
# and a Dirac in O, weighted by 1-pareto_weight.
pareto_weights = [1, 1, 1]
alpha_min = min(alphas)
best_arm = np.argmin(alphas)  # because smallest alpha and same C and pareto_weight

## Robust UCB parameters
u = 0  # threshold giving MAB formulation
eps = (alpha_min-1)/2  # eps such that 1+eps < alpha
def moment_max():
    # when same C on all arms
    return alpha_min/(alpha_min-1-eps)*Cs[0]**((1+eps)/alpha_min)
v = 2*moment_max()  # v such that: moment(1+eps) = alpha/(alpha-1-eps)*C^((1+eps)/alpha) <= v

regret_EH = np.zeros(n)
regret_RUCB = np.zeros(n)

for i in range(num_iter):
    # sample all rewards
    all_rewards = sample(n, alphas, scales, pareto_weights)

    ## Instantiation of the arms
    # for ExtremeHunter
    Arms_EH = {}
    for k in range(K):
        Arms_EH[k] = Arm(all_rewards[k, :], K=K, b=b, A=A, A_delta=A_delta)
    # for Robust UCB
    Arms_RUCB = {}
    for k in range(K):
        Arms_RUCB[k] = Arm(all_rewards[k, :], MAB=True, u=u, eps=eps, v=v)

    ## ExtremeHunter
    G_star = np.zeros(n)  # performance of best arm
    G_EH = 0  # performance of ExtremeHunter
    # Initialization
    for k in range(K):
        for t in range(N):
            Arms_EH[k].play()
            G_EH = max(G_EH, Arms_EH[k].last)
            G_star[k*N+t] = max(G_star[k*N+t-1], all_rewards[best_arm, k*N+t])
            regret_EH[k*N+t] += G_star[k*N+t] - G_EH
    # Run
    for t in range(K*N, n):
        winner = np.argmax([Arms_EH[k].B for k in range(K)])
        Arms_EH[winner].play()
        G_EH = max(G_EH, Arms_EH[winner].last)
        G_star[t] = max(G_star[t-1], all_rewards[best_arm, t])
        regret_EH[t] += G_star[t] - G_EH

    ## Robust UCB
    G_RUCB = 0  # performance of Robust UCB
    # Initialization
    for k in range(K):
        Arms_RUCB[k].play()
        G_RUCB = max(G_RUCB, Arms_RUCB[k].last)
        regret_RUCB[k] += G_star[k] - G_RUCB
    # Run
    for t in range(K+1, n+1):
        for k in range(K):
            # Truncated mean estimation
            Arms_RUCB[k].fmean_est(t**(-2))
            # Update Robust UCB index
            mean_est = Arms_RUCB[k].mean_est
            eps = Arms_RUCB[k].eps
            v = Arms_RUCB[k].v
            T = Arms_RUCB[k].T
            Arms_RUCB[k].B = mean_est + 4*v**(1/(1+eps))*(np.log(t**2)/T)**(eps/(1+eps))
        # play arm maximizing RUCB index
        winner = np.argmax([Arms_RUCB[k].B for k in range(K)])
        Arms_RUCB[winner].play()
        G_RUCB = max(G_RUCB, Arms_RUCB[winner].last)
        regret_RUCB[t-1] += G_star[t-1] - G_RUCB

    print 'finished:', i+1, '/', num_iter, 'experiments'

# divide by number of experiments
regret_EH = regret_EH/num_iter
regret_RUCB = regret_RUCB/num_iter

## Plot expected extreme regret
fig, ax = plt.subplots()
ax.plot(np.arange(1, n+1), regret_EH, 'k', label='ExtremeHunter')
ax.plot(np.arange(1, n+1), regret_RUCB, 'k--', label='Robust UCB')

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
plt.show()
