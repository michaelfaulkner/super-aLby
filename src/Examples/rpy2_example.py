
from math import *
import numpy as np
import matplotlib.pyplot as plt

# testing out the rpy2 library for random number generation
import rpy2.robjects.numpy2ri as n2ri
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector

# check that r packages are installed properly
packageNames = ('afex', 'lsmeans','normalp', 'LaplacesDemon')
if all(rpackages.isinstalled(x) for x in packageNames):
    have_packages = True
else:
   have_packages = False
if not have_packages:
    utils = rpackages.importr('utils')
    utils.chooseCRANmirror(ind=1)
    packnames_to_install = [x for x in packageNames if not rpackages.isinstalled(x)]
    if len(packnames_to_install) > 0:
        utils.install_packages(StrVector(packnames_to_install))

# import Psych dataset
#data = robjects.r('read.table(file = "http://personality-project.org/r/datasets/R.appendix3.data", header = T)')

# import afex package
#afex = rpackages.importr('afex')
#model = afex.aov_ez('Subject', 'Recall', data, within='Valence')
#print model

# import lsmeans and do pairwise contrast
#lsm  = rpackages.importr('lsmeans')
#pairwise = lsm.lsmeans(model, "Valence", contr="pairwise", adjust="holm")
#print pairwise

# import R sample functions for EP family
#MNM = rpackages.importr('MNM')
#samples = MNM.rmvpowerexp(n = 10, Location = 'c(0,0)', Beta = 1)
#print samples
#plt.hist(samples)
#plt.show()

#PEF = rpackages.importr('normalp') # p must be >= 1 in this version.
#samples_1 = PEF.rnormp(n = 1000, p = 1)
#samples_2 = PEF.rnormp(n = 1000, p = 3)
#print samples
#plt.hist(samples_1, color='red', normed=True)
#plt.hist(samples_2, color='blue', normed=True)
#plt.show()

r = robjects.r
n2ri.activate()

innov = r.rnorm(100)
x = np.zeros(100)
for i in xrange(99):
    x[i+1] = 0.9*x[i] + innov[i]
print r.mean(x)
Laplace = rpackages.importr('LaplacesDemon')
eff = Laplace.ESS(x)
print eff[0]

def ess(samples):
    return float(Laplace.ESS(samples)[0])
