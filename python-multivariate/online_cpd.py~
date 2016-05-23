from __future__ import division
import numpy as np
from scipy import stats
from scipy.special import gammaln

def online_cpd(data, hazard_func, observation_likelihood):
    maxes = np.zeros(len(data) + 1)
    flag = False
    cnt = 0
    change = False

    CP = np.zeros(1)
    last_cp = 0
    prev_cp = 0
    
    R = np.empty((len(data) + 1, len(data) + 1))
    R[0, 0] = 1
    
    for t, x in enumerate(data):
	curr_t = int(t-last_cp)
	
        # Evaluate the predictive distribution for the new datum under each of
        # the parameters.  This is the standard thing from Bayesian inference.
        predprobs = observation_likelihood.pdf(x)

        # Evaluate the hazard function for this interval
        H = hazard_func(np.array(range(curr_t+1)))
       
        # Evaluate the growth probabilities - shift the probabilities down and to
        # the right, scaled by the hazard function and the predictive
        # probabilities.
        R[1:curr_t+2, t+1] = R[0:curr_t+1, t] * predprobs * (1-H)
        
        # Evaluate the probability that there *was* a changepoint and we're
        # accumulating the mass back down at r = 0.
        R[0, t+1] = np.sum( R[0:curr_t+1, t] * predprobs * H)
        
        # Renormalize the run length probabilities for improved numerical
        # stability.
        R[:, t+1] = R[:, t+1] / np.sum(R[:, t+1])
    
        maxes[t] = R[:, t].argmax()
	
	if t > 0 and (maxes[t] - maxes[t-1]) < -20 :
	   flag = True
	   observation_likelihood.curr_theta()
	elif flag == True and t > 0 :
	   if abs(maxes[t] - maxes[t-1]) < 10 :
		cnt += 1
		if cnt > 10 :
		   change = True
		   flag = False
		   cnt = 0
		   CP = np.concatenate((CP, [t-maxes[t]+1]))
		   prev_cp = last_cp
		   last_cp = CP[-1]
		   curr_t = int(t-last_cp)
		   observation_likelihood.save_theta()
	   else :
		flag = False
		cnt = 0

	if change == True :
	   observation_likelihood.reset_theta(curr_t)
	   change = False

	# Update the parameter sets for each possible run length.
        observation_likelihood.update_theta(x)

    observation_likelihood.curr_theta()
    observation_likelihood.save_theta()
    theta = observation_likelihood.retrieve_theta()
    return R, maxes, CP, theta


def constant_hazard(r, lam):
    return 1/lam * np.ones(r.shape)


class StudentT:
    def __init__(self, alpha, beta, kappa, mu):
        self.alpha0 = self.alpha = np.array([alpha])
        self.beta0 = self.beta = np.array([beta])
        self.kappa0 = self.kappa = np.array([kappa])
        self.mu0 = self.mu = np.array([mu])

    def pdf(self, data):
        return stats.t.pdf(x=data, 
                           df=2*self.alpha,
                           loc=self.mu,
                           scale=np.sqrt(self.beta * (self.kappa+1) / (self.alpha *
                               self.kappa)))

    def update_theta(self, data):
        muT0 = np.concatenate((self.mu0, (self.kappa * self.mu + data) / (self.kappa + 1)))
        kappaT0 = np.concatenate((self.kappa0, self.kappa + 1.))
        alphaT0 = np.concatenate((self.alpha0, self.alpha + 0.5))
        betaT0 = np.concatenate((self.beta0, self.beta + (self.kappa * (data -
            self.mu)**2) / (2. * (self.kappa + 1.))))
            
        self.mu = muT0
        self.kappa = kappaT0
        self.alpha = alphaT0
        self.beta = betaT0


