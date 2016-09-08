from __future__ import division
import numpy as np

def sample(n, alphas, scales, pareto_weights):
    """
    Sample all rewards
    """
    K = len(alphas)
    all_rewards = np.zeros((K, n))
    for k in range(K):
        a = alphas[k]
        s = scales[k]
        p = pareto_weights[k]
        for t in range(n):
            if np.random.rand() < p:
                all_rewards[k, t] = (np.random.pareto(a)+1)*s
            else:
                all_rewards[k, t] = 0
    return all_rewards
