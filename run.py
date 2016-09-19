import numpy as np
import time
from arm import *
from sample import *

def f_run(args):

    begin = time.time()

    nb, n, K, N, alphas, scales, pareto_weights, b, A, A_delta, u, eps, v, best_arm, num_iter = args
    regret_EH = np.zeros(n)
    regret_RUCB = np.zeros(n)
    regret_rand = np.zeros(n)

    # sample all the rewards
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

    ## Random policy
    G_rand = 0
    for t in range(n):
        k = np.random.randint(K)  # choose random arm
        G_rand = max(G_rand, all_rewards[k, t])
        regret_rand[t] += G_star[t] - G_rand

    print 'done:', nb+1, '/', num_iter, 'in:', time.time()-begin, 'seconds'
    return [regret_EH, regret_RUCB, regret_rand]
