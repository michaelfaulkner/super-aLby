# super-aLby
super-aLby is a Python application that implements the super-relativistic Monte Carlo algorithm for the simulation of 
Bayesian probability models and classical *N*-body simulations of soft-matter models in statistical physics. For a 
closely connected discussion of kinetic-energy choice in Hamiltonian/hybrid Monte Carlo, see 
[\[Livingstone2019\]](https://academic.oup.com/biomet/article-abstract/106/2/303/5476364), where we first introduced 
super-relativistic Monte Carlo (though we did not name it).

## Installing

To install super-aLby, clone this repository.

The super-aLby Python application was written using cPython 3.8 but is likely to support any Python version >= 3.6 
(though we need to check this). super-aLby depends on `numpy`. Some of the sample-analysis code (i.e., scripts 
contained in the [`output`](src/output) directory) also depends on `matplotlib` and `rpy2`. 
[`markov_chain_diagnostics.py`](src/output/markov_chain_diagnostics.py) depends on the R package `LaplacesDemon`, which 
must be installed in R: download [the relevant binary](https://cran.r-project.org/web/packages/LaplacesDemon/) at CRAN 
and then enter `R CMD INSTALL <binary location>` in your terminal.

To manage external Python packages, we use conda environments via the 
[miniconda distribution](https://docs.conda.io/en/latest/miniconda.html). However, we found `rpy2` to be buggy when 
installed via conda. Instead, we `pip install rpy2` from within the project's conda environment.

## Using super-aLby

The user interface for each run of the super-aLby application consists of a configuration file that is an argument of 
the [`run.py`](src/run.py) script, which is located in the [`src`](src) directory. Configuration files should follow the 
[INI-file format](https://en.wikipedia.org/wiki/INI_file). To use super-aLby, go to the [`src`](src) directory and type 
`run.py <address of configuration file>`. The generated sample data will appear in the [`output`](src/output) directory 
(at an address given in the configuration file). Sample analysis can then be performed via scripts within the
[`output`](src/output) directory.

The [`run.py`](src/run.py) script expects the path to the configuration file as the first positional argument. The 
script also takes optional arguments. These are:
- `-h`, `--help`: Show the help message and exit.
- `-V`, `--version`: Show program's version number and exit.
- `-v`, `--verbose`: Increase verbosity of logging messages (multiple -v options increase the verbosity, the maximum is 
2).
- `-l LOGFILE`, `--logfile LOGFILE`: Specify the logging file. 

## Configuration files

A configuration file is composed of sections that correspond to either the [`run.py`](src/run.py) file, the model 
settings (contained in [`model_settings/__init__.py`](src/model_settings/__init__.py)), or a class of the super-aLby 
application. The only required sections for the run script are those corresponding to the [`run.py`](src/run.py) file 
and the model settings:

```INI
[Run]
mediator = some_mediator
```

and (for example)

```INI
[ModelSettings]
beta = 1.0
number_of_particles = 2
size_of_particle_space = None
range_of_initial_particle_positions = 1.0
```

`some_mediator` corresponds the mediator used (for this particular simulation) in the `run.py` file. The mediator 
serves as the central hub in the application and also hosts the Markov process. The two possible mediators are 
[`leapfrog_mediator`](src/mediator/leapfrog_mediator.py), which implements the simulation using the leapfrog numerical 
integrator, and [`toroidal_leapfrog_mediator`](src/mediator/toroidal_leapfrog_mediator.py), which differs from 
[`leapfrog_mediator`](src/mediator/leapfrog_mediator.py) in that it corrects particle positions to account for 
periodic boundary conditions (i.e., for particles existing on a toroidal space) after each numerical integration step.
We use [`leapfrog_mediator`](src/mediator/leapfrog_mediator.py) for models on Euclidean space and 
[`toroidal_leapfrog_mediator`](src/mediator/toroidal_leapfrog_mediator.py) for models on compact subspaces of Euclidean 
space (with toroidal geometry).

The ```[ModelSettings]``` section specifies the *NVT* physical parameters of the simulation and the range of the 
initial particle positions. `beta` is a `float` that represents the inverse temperature. `number_of_particles` is an 
`int` that represents number of particles. `size_of_particle_space` is either `None`, a `float` or a Python `list` of 
`None` or `float` values, each of which would represent the size and dimensions of the spaces on which each particle 
exits (`None` corresponds to the whole real line). `range_of_initial_particle_positions` is either a `float`, a Python 
`list` of `float` values  or a Python `list` of `float` values, each of which would represent the range of the initial 
positions of each particle. The above example represents a two-particle system at `beta = 1.0`, where each particle 
exists on the entire real line and has initial position *1.0*, while

```INI
[ModelSettings]
beta = 2.0
number_of_particles = 4
size_of_particle_space = [1.0, 1.0]
range_of_initial_particle_positions = [[-0.5, 0.5], [-0.5, 0.5]]
```

represents a four-particle system at `beta = 2.0`, where each particle exists on the toroidal compact subspace (of 
volume *1.0 x 1.0*) of two-dimensional Euclidean space and takes an initial position anywhere on that subspace.

The following sections of the configuration file choose the parameters in the `__init__` methods of the mediator. Each 
section contains pairs of properties and values. The property corresponds to the name of the argument in the `__init__` 
method of the given class, and its value provides the arguments. Properties and values should be given in snake_case; 
sections should be given in CamelCase. If a value corresponds to the instance of another class, e.g. `potential = 
exponential_power_potential`, then a corresponding section is required, e.g., `[ExponentialPowerPotential]`.

Some example configuration files are located in the [`src/config_files`](src/config_files) directory.
