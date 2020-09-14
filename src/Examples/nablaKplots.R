# Plots of different \nabla K(p) for different choices of kinetic energy

library(normalp) # exponential power family
library(ars) # adaptive rejection sampling for relativistic family
library(rmutil) # laplace distribution


n = 1000000

# Gaussian choice
p_gauss = rnorm(n)

nabla_gauss = p_gauss

# Laplace
p_laplace = rlaplace(n)

nabla_laplace = p_laplace / abs(p_laplace)

# Exponential power
p_PE43 = rnormp(n, p = 4/3)

nabla_PE43 = (p_PE43 / abs(p_PE43)) * abs(p_PE43)^(1/3)

p_PE4 = rnormp(n, p = 4)

nabla_PE4 = p_PE4^3

domain = c(-2,2)

?plotmath
install.packages('tikzDevice',repos='http://www.rforge.net/') 
require(tikzDevice)

#tikz('test.tex', standAlone = TRUE, width=5, height=5) # used for tikzDevice
font = 1
par(mfrow = c(1,4), cex.main = font, cex.lab = font, cex.axis = font)
hist(nabla_laplace, xlim = domain, freq = F,
     col = "gray", main = "Laplace", xlab = expression(nabla * K(p)), ylab = "Density")
hist(nabla_PE43, xlim = domain, breaks = 30, freq = F,
     col = "gray", main = "Exponential Power 4/3", xlab = expression(nabla * K(p)), ylab = "Density")
hist(nabla_gauss, xlim = domain, breaks = 50, freq = F,
     col = "gray", main = "Gaussian", xlab = expression(nabla * K(p)), ylab = "Density")
hist(nabla_PE4, xlim = domain, breaks = 150, freq = F,
     col = "gray", main = "Exponential Power 4", xlab = expression(nabla * K(p)), ylab = "Density")

# dev.off() # used for tikzDevice