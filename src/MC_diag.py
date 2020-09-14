from math import *
import numpy as np

# Call rpy2 libraries
import rpy2.robjects.numpy2ri as n2ri
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector

r = robjects.r # make r easy to access
n2ri.activate() # activate numpy R interface

Laplace = rpackages.importr('LaplacesDemon') # import Laplaces Demon package

def ess(samples):
    return Laplace.ESS(samples)

def thin(samples, level):
    n = len(samples)
    index = np.asarray(range(n))
    index_to_keep = np.nonzero( index/level==index/float(level) )
    return samples[index_to_keep]