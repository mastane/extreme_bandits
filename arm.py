from __future__ import division
import numpy as np
from useful_functions import *

# Arm
class Arm:

    def __init__(self, rewards, MAB=False, K=3, b=1, A=2e-3, A_delta=1, u=None, eps=None, v=None):
        # time horizon
        self.n = len(rewards)
        # number of arms
        self.K = K
        # MAB==False --> ExtremeHunter, MAB==True --> Robust UCB
        self.MAB = MAB
        # all rewards (t=1, ..., n) for this arm
        self.rewards = rewards
        # number of times arm has been pulled
        self.T = 0
        # index
        self.B = 0
        # all samples drawn from this arm (if MAB==True: truncated rewards for Robust UCB)
        self.samples = np.zeros(self.n)
        # last sample drawn from this arm
        self.last = 0
        if not self.MAB:  # ExtremeHunter algorithm
            # ExtremeHunter parameters
            self.b = b
            self.A = A  # constant used to compute N
            self.N = int(self.A*np.log(self.n)**(2*(2*self.b+1)/self.b))
            self.delta = np.exp(-np.log(self.n)**2)/(2*self.K*self.n)
            self.A_delta = A_delta  # constant used to estimate 1/alpha
            # estimators of 1/alpha and C
            self.h, self.C_est = 0, 0
            # useful quantities
            self.B1, self.B2 = 0, 0
            # other constants
            self.D, self.E = 1, 1
        else:  # Robust UCB algorithm
            # Robust UCB parameters
            self.u = u  # threshold for rewards
            self.eps = eps  # such that 1+eps < min(alphas)
            self.v = v  # such that moments(1+eps) <= v
            self.last_trunc = 0  # last reward truncated with threshold u
            # Truncated mean estimator
            self.mean_est = 0


    def play(self):
        """
        draw a new sample and update index
        """
        if not self.MAB:  # ExtremeHunter
            # Update T
            self.T += 1
            # Draw and store new sample
            self.last = self.rewards[self.T-1]
            self.samples[self.T-1] = self.last
            if self.T >= self.N:
                # Update estimators
                # self.h = EstimateH(self.samples[:self.T], self.n, self.delta, self.A_delta)
                self.h = EstimateH_rough(self.samples[:self.T])
                self.C_est = EstimateC(self.samples[:self.T], self.h, self.b)
                # Compute useful quantities
                self.B1 = self.D*np.sqrt(np.log(1/self.delta))*self.T**(-self.b/(2*self.b+1))
                self.B2 = self.E*np.sqrt(np.log(self.T/self.delta))*np.log(self.T)*self.T**(-self.b/(2*self.b+1))
                # Update index B
                self.B = ((self.C_est+self.B2)*self.n)**(self.h+self.B1)*Gamma(self.h, self.B1)
        else:  # Robust UCB
            # Update T
            self.T += 1
            # Draw and store new sample
            self.last = self.rewards[self.T-1]
            if self.last > self.u:
                self.last_trunc = self.last
            else:
                self.last_trunc = 0
            self.samples[self.T-1] = self.last_trunc

    def fmean_est(self, conf):
        """
        truncated mean estimator
        """
        res = 0
        for t in range(self.T):
            x = self.samples[t]
            if x <= (self.v*t/np.log(1/conf))**(1/(1+self.eps)):
                res += x
        self.mean_est = res/self.T
