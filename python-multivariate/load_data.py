import numpy as np
import scipy.io as sio

def load_data(filename):
	return sio.loadmat(filename)

