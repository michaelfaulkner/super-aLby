# Kinetic energies library
from math import *
import numpy as np
import sys
import adaptive_rejection_sampling as AdaptReject

# import R sample functions for EP family
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
import rpy2.robjects.numpy2ri as n2ri
#from rpy2.rinterface import rternalize

r = robjects.r # make r easy to access
n2ri.activate() # activate numpy R interface

Laplace = rpackages.importr('LaplacesDemon') # import Laplaces Demon package
ARS = rpackages.importr('ars') # import adaptive rejection sampling package
PEF = rpackages.importr('normalp') # import exponential power family package

#@ri.rternalize # allow python functions to work within R functions

# Global functions
def norm(p):
    if type(p) is float or type(p) is int:
        ans = sqrt(p**2)
    else:
        ans = sqrt(np.dot(p,p))
    return ans

# Gaussian (multivariate)
class Gauss:
    """The Gaussian family"""
    #def __init__(self, cov):
    #    self.cov = cov

    def value(self, p):
        #cov = self.cov
        # INSERT do we need multivariate?
        return np.dot(p,p) / 2.0

    def grad(self, p):
        #cov = self.cov
        return p

    def sample(self, n):
        #cov = self.cov # currently not used
        mean = np.zeros(n)
        cov = np.identity(n)
        return np.random.normal(size = n)

# Laplace (independent)
class Laplace:
    """The Laplace family"""
    def value(self, p):
        return sum(np.absolute(p))

    def grad(self, p):
        return np.sign(p)

    def sample(self, n):
        return np.random.laplace(size = n)

# General Power Exponential family (univariate at present) NEEDS FIXING
class PE_family:
    """The power exponential family"""
    def __init__(self, beta):
        self.beta = beta

    def value(self, p):
        b = self.beta
        return sum(np.absolute(p)**b) / float(b)

    def grad(self, p):
        b = self.beta
        return np.absolute(p)**(b-1) * np.sign(p)

    def sample(self, n):
        b = self.beta
        return np.asarray(PEF.rnormp(n, p = b))

# Student (independent)
class Student:
    """The Student's t family"""
    def __init__(self, nu):
        self.nu = nu # degrees of freedom

    def value(self, p):
        nu = self.nu
        return sum((nu+1)/2.0 * np.log(1 + p**2/float(nu)))

    def grad(self, p):
        nu = self.nu
        return (nu + 1)*p / (nu + p**2)

    def sample(self, n):
        nu = self.nu
        return np.random.standard_t(df = nu, size = n)


# Relativistic kinetic energies
class GR_family:
    """Generalised relativistic kinetic energy"""
    def __init__(self, beta, c = 1, m = 1):
        self.beta = beta
        self.m = m
        self.c = c

    def value(self, p):
        beta = self.beta
        m = self.m
        c = self.c

        z = p**2/float(m**2 * c**2)
        c1 = float(m)**beta * float(c)**(1+beta) / beta
        return sum( c1 * (z + 1) ** (beta/2.0) )

    def __logpi(self, p): # private function negating U.value
        beta = self.beta
        m = self.m
        c = self.c

        z = p**2/float(m**2 * c**2)
        c1 = float(m)**beta * float(c)**(1+beta) / beta
        return - c1 * (z + 1) ** (beta/2.0)

    def grad(self, p):
        beta = self.beta
        m = self.m
        c = self.c

        z = p**2/float(m**2 * c**2)
        c2 = float(m)**(beta-2.0) * float(c)**(beta - 1.0)
        return c2 * (z+1) ** (beta/2.0 - 1) * p

    def __dlogpi(self, p): # private function negating U.grad univariate
        beta = self.beta
        m = self.m
        c = self.c

        z = p**2/float(m**2 * c**2)
        c2 = float(m)**(beta-2.0) * float(c)**(beta - 1.0)
        return - c2 * (z+1) ** (beta/2.0 - 1) * p

    def sample(self, n):
        # call adaptive rejection sampler

        # prototype from r (might be faster, if not can interface c)
        #rlogpi = rternalize(self.__logpi)
        #rdlogpi = rternalize(self.__dlogpi)

        #sample = ARS.ars(n, rlogpi, rdlogpi)

        ars = AdaptReject.AdaptiveRejectionSampling(self.__logpi, self.__dlogpi)

        return ars.draw(n)
    
# Power Relativistic kinetic energies
class PR_family:
    """Power relativistic kinetic energy"""
    def __init__(self, beta, gamma = 1):
        self.beta = beta
        self.gamma = gamma

    def value(self, p):
        beta = self.beta
        gamma = self.gamma
        
        vec = (1+ p**2/gamma)**(beta/2.0) / float(beta)
        return sum(vec)

    def __logpi(self, p): # private function negating U.value
        beta = self.beta
        gamma = self.gamma
        
        z = (1 + p**2/float(gamma))
        vec = z ** (beta/2.0) / float(beta)
        return -sum(vec)

    def grad(self, p):
        beta = self.beta
        gamma = self.gamma
        
        return (p/float(gamma)) * (1 + p**2/float(gamma))**(beta/2.0 - 1)

    def __dlogpi(self, p): # private function negating U.grad univariate
        beta = self.beta
        gamma = self.gamma
        
        z = (1 + p**2/float(gamma))
        return  - (p/float(gamma)) * z ** (beta/2.0 - 1)

    def sample(self, n):
        # call adaptive rejection sampler

        # prototype from r (might be faster, if not can interface c)
        #rlogpi = rternalize(self.__logpi)
        #rdlogpi = rternalize(self.__dlogpi)

        #sample = ARS.ars(n, rlogpi, rdlogpi)

        ars = AdaptReject.AdaptiveRejectionSampling(self.__logpi, self.__dlogpi)

        return ars.draw(n)
