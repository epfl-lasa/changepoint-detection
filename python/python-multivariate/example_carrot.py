from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import seaborn

import cProfile
import StudentTMulti as st
import online_cpd as oncpd
import generate_data as gd
import load_data as ld
import scipy.io as sio
from functools import partial

if __name__ == '__main__':
  show_plot = True
  dim = 2
  if dim == 1:
    partition, data = gd.generate_normal_time_series(7, 50, 200)
    prior = oncpd.StudentT(alpha=1, beta=1, kappa=1, mu=0)
    changes = np.cumsum(partition)
  else:
    #partition, data = gd.generate_multinormal_time_series(5, dim, 100, 300)
    data_mat = ld.load_data("CarrotGrating")
    X = data_mat["Xn"]
    data = np.transpose(X[0,0])
    dim = data.shape[1]
    prior = st.StudentTMulti(dim)
    changes = 0

  if show_plot:
    fig, ax = plt.subplots(figsize=[16,12])
    #for p in changes:
    #  ax.plot([p,p],[np.min(data),np.max(data)],'r')
    for d in range(dim):
      ax.plot(data[:,d])
    plt.show()

  print "Starting detection algorithm"
  R, maxes, CP, theta = oncpd.online_cpd(data,partial(oncpd.constant_hazard,lam=200),prior)

  print "Changepoints locations:"
  print CP
  print "Segment parameters:"
  print theta

  if show_plot:
    fig, ax = plt.subplots(figsize=[18, 16])
    ax = fig.add_subplot(2, 1, 1)
    #if changes != 0:
    #  for p in changes:
    #     ax.plot([p,p],[np.min(data),np.max(data)],'r')
    #else :
    for p in CP:
      ax.plot([p,p],[np.min(data),np.max(data)],'r')
    for d in range(dim):
      ax.plot(data[:,d])
    ax = fig.add_subplot(2, 1, 2, sharex=ax)
    ax.plot(maxes)
    plt.show()

