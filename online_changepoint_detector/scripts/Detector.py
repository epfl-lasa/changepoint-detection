from __future__ import division
import rospy
from rospy.numpy_msg import numpy_msg
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

class Detector:
    def __init__(self):
	self.theta = 0
	self.CP = np.zeros(1)
	self.R_old = [1]
	self.maxes = []
	self.curr_t = 0
	self.flag = False
	self.cnt = 0
	self.change = False
	self.prev_cp = 0
	self.last_cp = 0

    def detect(self, x, hazard_func, observation_likelihood):
	
	t = self.curr_t
	R = np.empty(t+2)

	predprobs = observation_likelihood.pdf(x)
	#if len(predprobs) == 1:
	#   predprobs = [predprobs]

        # Evaluate the hazard function for this interval
        H = hazard_func(np.array(range(self.curr_t+1)))
       
        # Evaluate the growth probabilities
        R[1:t+2] = self.R_old[0:t+1] * predprobs * (1-H)
        #np.dot(self.R_old[0:t+1],predprobs) * (1-H)
	
        # Evaluate the probability that there *was* a changepoint and we're
        # accumulating the mass back down at r = 0.
        R[0] = np.sum( self.R_old[0:t+1] * predprobs * H)
        
        # Renormalize the run length probabilities for improved numerical
        # stability.
	#self.R = np.empty(t+1)
        self.R_old = R / np.sum(R)
    
        self.maxes = np.append(self.maxes, self.R_old.argmax())
	
	if t > 0 and (self.maxes[-1] - self.maxes[-2]) < -20 :
	   self.flag = True
	   observation_likelihood.curr_theta()
	elif self.flag == True and t > 0 :
	   if abs(self.maxes[-1] - self.maxes[-2]) < 10 :
		self.cnt += 1
		if self.cnt > 10 :
		   self.change = True
		   self.flag = False
		   self.cnt = 0
		   self.CP = np.concatenate((self.CP, [self.last_cp+t-self.maxes[-1]+1]))
		   self.displayCP()
		   self.prev_cp = self.last_cp
		   self.last_cp = self.CP[-1]
		   self.curr_t = int(t-(self.last_cp-self.prev_cp))
		   observation_likelihood.save_theta()
	   else :
		self.flag = False
		self.cnt = 0

	if self.change == True :
	   observation_likelihood.reset_theta(self.curr_t)
	   self.change = False

	# Update the parameter sets for each possible run length.
        observation_likelihood.update_theta(x)

	self.curr_t += 1

    def retrieve(self, observation_likelihood):

        observation_likelihood.curr_theta()
        observation_likelihood.save_theta()
        self.theta = observation_likelihood.retrieve_theta()
        return self.maxes, self.CP, self.theta

    def displayCP(self):
	
	rospy.loginfo("\nLast changepoint fount at position %f\n", self.CP[-1])

    def plot_data_CP(self, x):

	plt.scatter(len(self.maxes)*np.ones(len(x)), x)
	plt.plot([self.CP,self.CP],[np.min(x),np.max(x)],'r')
	plt.pause(0.0001)
