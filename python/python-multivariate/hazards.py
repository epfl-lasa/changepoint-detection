from __future__ import division
import numpy as np

def constant_hazard(r, lam):
    return 1/lam * np.ones(r.shape)
