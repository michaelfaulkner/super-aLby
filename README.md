# super-aLby
super-aLby is a Python application that implements the super-relativistic Monte Carlo algorithm for the simulation of 
Bayesian probability models and classical *N*-body soft-matter models in statistical physics. For a closely connected 
discussion of kinetic-energy choice in Hamiltonian/hybrid Monte Carlo, see [\[Livingstone2019\]](
https://academic.oup.com/biomet/article-abstract/106/2/303/5476364), where we first introduced super-relativistic Monte 
Carlo (though we did not name it).

## Installation

To install super-aLby, clone this repository.

super-aLby was written using Python 3.8 but is likely to support any Python version >= 3.6 (though we need to check 
this). It has been tested with CPython.

super-aLby depends on [`numpy`](https://numpy.org). Some of the sample-analysis code (i.e., scripts contained in the 
[`output`](src/output) directory) also depends on [`matplotlib`](https://matplotlib.org) and [`rpy2`](
https://rpy2.github.io). [`markov_chain_diagnostics.py`](src/output/markov_chain_diagnostics.py) depends on the R 
package [`LaplacesDemon`](https://cran.r-project.org/web/packages/LaplacesDemon/), which must be installed in R: 
download [the relevant binary](https://cran.r-project.org/web/packages/LaplacesDemon/) at CRAN and then run `R CMD 
INSTALL <binary location>` in your terminal.

To manage external Python packages, we use [conda](https://docs.conda.io/projects/conda/en/latest/) environments via 
the [miniconda distribution](https://docs.conda.io/en/latest/miniconda.html). However, we found [`rpy2`](
https://rpy2.github.io) to be buggy when installed via conda. Instead, we `pip install rpy2` from within the project's 
conda environment (after having `conda install`ed [`numpy`](https://numpy.org) and [`matplotlib`](
https://matplotlib.org)).

## Implementation

The user interface of the super-aLby application consists of the [`run.py`](src/run.py) script and a configuration 
file. The [`run.py`](src/run.py) script expects the path to the configuration file as the first positional argument. 
Configuration files should be located in the [`config_files`](src/config_files) directory and follow the [INI-file 
format](https://en.wikipedia.org/wiki/INI_file). The [`run.py`](src/run.py) script is located in the [`src`](src) 
directory. 

To run the super-aLby application, open your terminal, navigate to the [`src`](src) directory and enter `python run.py 
<configuration file>`. The generated sample data will appear in the [`output`](src/output) directory (at a location 
given in the configuration file). Sample analysis can then be performed via scripts within the [`output`](src/output) 
directory.

The [`run.py`](src/run.py) script also takes optional arguments. These are:
- `-h`, `--help`: Show the help message and exit.
- `-V`, `--version`: Show program's version number and exit.
- `-v`, `--verbose`: Increase verbosity of logging messages (multiple -v options increase the verbosity, the maximum is 
2).
- `-l LOGFILE`, `--logfile LOGFILE`: Specify the logging file. 

## Configuration files

A configuration file is composed of sections that correspond to either the [`run.py`](src/run.py) file, the model 
settings (contained in [`model_settings/__init__.py`](src/model_settings/__init__.py)), or a class of the super-aLby 
application. Each configuration file must contain `[Run]` and `[ModelSettings]` sections, which (respectively) 
correspond to the [`run.py`](src/run.py) file and the [model settings](src/model_settings/__init__.py):

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
serves as the central hub of the application and also hosts the Markov process. The three possible mediators are 
[`euclidean_leapfrog_mediator`](src/mediator/euclidean_leapfrog_mediator.py), which implements the simulation using the 
leapfrog numerical integrator, [`toroidal_leapfrog_mediator`](src/mediator/toroidal_leapfrog_mediator.py), which 
differs from [`euclidean_leapfrog_mediator`](src/mediator/euclidean_leapfrog_mediator.py) in that it corrects particle 
positions to account for periodic boundary conditions (i.e., for particles existing on a toroidal space) after each 
numerical integration step, and [`lazy_toroidal_leapfrog_mediator`](src/mediator/lazy_toroidal_leapfrog_mediator.py), 
which differs from [`toroidal_leapfrog_mediator`](src/mediator/toroidal_leapfrog_mediator.py) in that it corrects 
particle positions only at the end of each leapfrog trajectory. We use [`euclidean_leapfrog_mediator`](
src/mediator/euclidean_leapfrog_mediator.py) for models on Euclidean space and [`toroidal_leapfrog_mediator`](
src/mediator/toroidal_leapfrog_mediator.py) or [`lazy_toroidal_leapfrog_mediator`](
src/mediator/lazy_toroidal_leapfrog_mediator.py) for models on compact subspaces of Euclidean space (with toroidal 
geometry).

The ```[ModelSettings]``` section specifies both the *NVT* physical parameters of the simulation and the range of the 
initial particle positions. `beta` is a `float` that represents the inverse temperature. `number_of_particles` is an 
`int` that represents number of particles. `size_of_particle_space` represents the size and dimensions of the spaces on 
which each particle exits and is either `None`, a `float` or a Python `list` of `None` or `float` values (`None` 
corresponds to the whole real line). `range_of_initial_particle_positions` represents the range of the initial position 
of each particle and is either a `float`, a one-dimensional Python `list` of length 
`len(range_of_initial_particle_positions)` and composed of `float` values, or a two-dimensional Python `list` of size 
`(len(range_of_initial_particle_positions), 2)` and composed of `float` values. The above example represents a 
two-particle system at `beta = 1.0`, where each particle exists on the entire real line and has initial position *1.0*, 
while

```INI
[ModelSettings]
beta = 2.0
number_of_particles = 4
size_of_particle_space = [1.0, 1.0]
range_of_initial_particle_positions = [[-0.5, 0.5], [-0.5, 0.5]]
```

represents a four-particle system at `beta = 2.0`, where each particle exists on the toroidal compact subspace (of 
volume *1.0 x 1.0*) of two-dimensional Euclidean space and takes an initial position anywhere on that subspace.

The remaining sections of the configuration file correspond to the different classes chosen for the simulation. Each 
section contains pairs of properties and values. The property corresponds to the name of the argument in the `__init__` 
method of the corresponding class, and its value provides the argument. Property-value pairs must be provided for all 
properties for which the class does not provide a default value; for each property that does not have a default value, 
a property-value pair may be given. Properties and values should be given in snake_case; sections should be given in 
CamelCase. If a value corresponds to the instance of another class, then a corresponding section is required. In our 
example, the first of the remaining sections is therefore either of the form

```INI
[SomeMediator]
potential = some_potential
sampler = some_sampler
kinetic_energy = some_kinetic_energy
...
```

or of the form

```INI
[SomeMediator]
potential = some_potential
sampler = some_sampler
noise_distribution = some_noise_distribution
...
```

where the ellipsis accounts for further pairs of properties and values that do not correspond to other classes. The 
first / second example therefore also requires the sections `[SomePotential]`, `[SomeSampler]` and`[SomeKineticEnergy]` 
/ `[SomeNoiseDistribution]`. 

Some example configuration files are located in the [`src/config_files`](src/config_files) directory. To get a feel for 
the application, run `python run.py 
config_files/convergence_tests/exponential_power_potential_power_equals_4/super_relativistic_kinetic_energy.ini`, 
before running `python sample_analysis/test_convergence.py 
config_files/convergence_tests/exponential_power_potential_power_equals_4/super_relativistic_kinetic_energy.ini` 
once the simulation has finished. 
