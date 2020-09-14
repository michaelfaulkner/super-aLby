# Potentials

# Gaussian family
def U_gauss(x):
    return np.dot(x,x) / 2.0

def dU_gauss(x):
    return x

# Funnel
def U_funnel(x):
    return x[1]**2 + x[2]**2 + (x[1]**2)*(x[2]**2)

def dU_funnel(x):
    grad = np.empty(2)
    grad[1] = 2*x[1] * (1+x[2]**2)
    grad[2] = 2*x[2] * (1+x[1]**2)
    return grad
