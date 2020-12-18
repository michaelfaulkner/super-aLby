# super-aLby
super-aLby is a Python application for super-relativistic Monte Carlo simulation of Bayesian and soft-matter models.

The main code of super-aLby (i.e., that which runs via ```python run.py <config file>```) requires the installation of 
the following non-standard Python libraries: numpy.

The sample-analysis code of super-aLby requires the installation of the following non-standard Python libraries: numpy, 
rpy2 and matplotlib. The R package LaplacesDemon must also be installed in R, which can be achieved by downloading [the 
relevant binary](https://cran.r-project.org/web/packages/LaplacesDemon/) at CRAN and then entering ```R CMD INSTALL 
<binary location>``` in terminal.

To manage our external Python packages, we use conda environments via the [miniconda 
distribution](https://docs.conda.io/en/latest/miniconda.html). However, we found rpy2 to be buggy when installed via 
conda. Instead, we ```pip install rpy2``` from within the project's conda environment. 