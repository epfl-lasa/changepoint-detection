from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import seaborn

import cProfile
import StudentTMulti as st
import Detector as dt
import hazards as hz
import generate_data as gd
from functools import partial

if __name__ == '__main__':
  show_plot = False
  dim = 2
  partition, data = gd.generate_multinormal_time_series(5, dim, 100, 300)
  prior = st.StudentTMulti(dim)
  changes = np.cumsum(partition)

  if show_plot:
    fig, ax = plt.subplots(figsize=[16,12])
    for p in changes:
      ax.plot([p,p],[np.min(data),np.max(data)],'r')
    for d in range(dim):
      ax.plot(data[:,d])
    plt.show()

  detector = dt.Detector()

  plt.axis([0, len(data), np.min(data), np.max(data)])
  plt.ion()

  for t, x in enumerate(data):
    detector.detect(x,partial(hz.constant_hazard,lam=200),prior)
    detector.plot_data_CP(x)

  maxes, CP, theta = detector.retrieve(prior)

  print "Changepoints locations:"
  print CP
  print "Segment parameters:"
  print theta

  if show_plot:
    fig, ax = plt.subplots(figsize=[18, 16])
    ax = fig.add_subplot(2, 1, 1)
    for p in CP:
      ax.plot([p,p],[np.min(data),np.max(data)],'r')
    for d in range(dim):
      ax.plot(data[:,d])
    ax = fig.add_subplot(2, 1, 2, sharex=ax)
    ax.plot(maxes)
    plt.show()

