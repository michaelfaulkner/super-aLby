#install.packages(c('repr', 'IRdisplay', 'evaluate', 'crayon', 'pbdZMQ', 'devtools', 'uuid', 'digest'))
#devtools::install_github('IRkernel/IRkernel')
# Don’t forget step 2/2!
#IRkernel::installspec()

install.packages('LaplacesDemon')
x = rep(NA, 100)
x[1] = 0
for (i in 1:99) {
  x[i+1] = 0.9*x[i] + rnorm(1)
}
x
library(LaplacesDemon)
ESS(x)

install.packages('ars')
library(ars)
?ars
U = function(p) {
  return ( -(p^2 + 1)^(1.5/2) )
}

dU = function(p) {
  return ( -1.5 * ( p^2 + 1)^(-1/4) * p )
}

samples = ars(n=1000,f = U,fprima = dU)

hist(samples, freq = F)
density = function(p) { exp(U(p)) }
Z = integrate(density, lower = -Inf, upper = Inf)$value
ndensity = function(p) { exp(U(p))/Z }

curve(ndensity, from = -4, to = 4, add = T)
