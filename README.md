# super-aLby
super-aLby is a Python application that implements various Monte Carlo sampling algorithms for both classical *N*-body 
models in statistical physics (including Wick-rotated quantum actions) and Bayesian probability models.  The Monte Carlo
algorithms include the Metropolis, event-chain, super-relativistic, Hamiltonian, Wolff and Swendsen-Wang Monte Carlo 
algorithms.

For a closely connected discussion of kinetic-energy choice in Hamiltonian/hybrid Monte Carlo, see 
[\[Livingstone2019\]](https://academic.oup.com/biomet/article-abstract/106/2/303/5476364), where we first introduced 
super-relativistic Monte Carlo (though we did not name it).  super-aLby in fact started life as an application for 
Hamiltonian and super-relativistic Monte Carlo (hence the name super-aLby, in reference to Einstein).

## Installation

To install super-aLby, clone this repository.

super-aLby was written using Python 3.8 but is likely to support any Python version >= 3.6 (though we need to check 
this). It has been tested with CPython.

super-aLby depends on [`numpy`](https://numpy.org). Some of the sample-analysis code (i.e. scripts contained in the 
[`sample_analysis`](src/sample_analysis) directory) also depends on [`matplotlib`](https://matplotlib.org).

## Implementation

The user interface of the super-aLby application consists of the [`run.py`](src/run.py) script and a configuration 
file. The [`run.py`](src/run.py) script expects the path to the configuration file as the first positional argument. 
Configuration files should be located in the [`config_files`](src/config_files) directory and follow the [INI-file 
format](https://en.wikipedia.org/wiki/INI_file). The [`run.py`](src/run.py) script is located in the [`src`](src) 
directory. 

To run the super-aLby application, open your terminal, navigate to the [`src`](src) directory and enter `python run.py 
<configuration file>`. The generated sample data should then appear in the [`output`](src/output) directory (at a 
location given in the configuration file). Sample analysis can then be performed via scripts within the 
[`sample_analysis`](src/sample_analysis) directory.

The [`run.py`](src/run.py) script also takes optional arguments. These are:
- `-h`, `--help`: Show the help message and exit.
- `-V`, `--version`: Show program's version number and exit.

## Configuration files

A configuration file is composed of sections that correspond to either the [`run.py`](src/run.py) file, the model 
settings (contained in [`model_settings/__init__.py`](src/model_settings/__init__.py)), or a class of the super-aLby 
application. Each configuration file must contain `[Run]` and `[ModelSettings]` sections, which (respectively) 
correspond to the [`run.py`](src/run.py) file and the [model settings](src/model_settings/__init__.py).  For example, 

```INI
[Run]
mediator = some_mediator
number_of_jobs = 1
max_number_of_cpus = 1
```

and 

```INI
[ModelSettings]
number_of_particles = 2
size_of_particle_space = None
range_of_initial_particle_positions = 1.0
```

### The mediator

In the above `[Run]` section, `some_mediator` corresponds to the mediator used (for this particular simulation) in the `run.py` file. The mediator 
serves as the central hub of the application and also hosts the Markov process. We provide multiple mediators in the 
[`mediator`](src/mediator) package:

1. [`event_chain_mediator`](src/mediator/event_chain_mediator.py) implements event-chain Monte Carlo.
2. [`metropolis_mediator`](src/mediator/metropolis_mediator.py) implements Metropolis Monte Carlo.
3. [`swendsen_wang_mediator`](src/mediator/swendsen_wang_mediator.py) implements Swendsen-Wang Monte Carlo for the 2D Ising model.
4. [`wolff_mediator`](src/mediator/wolff_mediator.py) implements Wolff Monte Carlo for the 2D Ising model.
5. [`unbounded_leapfrog_mediator`](src/mediator/unbounded_leapfrog_mediator.py), 
[`toroidal_leapfrog_mediator`](src/mediator/toroidal_leapfrog_mediator.py) and 
[`lazy_toroidal_leapfrog_mediator`](src/mediator/lazy_toroidal_leapfrog_mediator.py) implement Hamiltonian Monte Carlo 
with a leapfrog integrator (the first is for models defined on unbounded Euclidean space; the second is for models 
defined on any torus; the third is for models defined on any torus (but while only correcting for periodic boundaries 
at each Metropolis step, hence _lazy_)).

(Note that any reference to a torus is to the [flat torus](https://en.wikipedia.org/wiki/Torus#Flat_torus).)

Additional mediators are present in the [`mediator`](src/mediator) package.  Each is an abstract parent class from 
which one of the above mediators inherits.  We use this same inheritance structure in the following packages:

1. [`kinetic_energy`](src/kinetic_energy) which provides functionality for various kinetic energies used in Hamiltonian Monte Carlo.
2. [`noise_distribution`](src/noise_distribution) which provides functionality for various noise distributions used in Metropolis Monte 
Carlo.
3. [`potential`](src/potential) which provides functionality for the potential (energy) function that defines the model.
4. [`sampler`](src/sampler) which provides functionality for sampling various system observables.

The mediator interacts with (instances of) classes contained in each of these packages, such that the classes never 
interact with each other.  In addition, some [`potential`](src/potential) classes instantiate classes contained in the 
[`linked_lists`](src/linked_lists) package.  This provides linked-list functionality for cell-based evaluation of certain 
multi-particle models defined on the two- or three-dimensional torus.  Below we detail how the configuration file 
chooses the classes that will be used.

### Rest of [Run] section

`number_of_jobs` and `max_number_of_cpus` are `int` values and should also be specified in the `[Run]` section. They 
correspond, respectively, to the number of independent realisations of the same process (i.e. simulation) and the 
maximum number of CPUs that should be simultaneously used for each of these realisations (to avoid overloading personal 
machines). 

### Model settings

The ```[ModelSettings]``` section specifies some global model parameters and the possible initial particle positions:

- `number_of_particles` is an `int` that represents the number of particles.  For any model that is not a Wick-rotated 
quantum action, this should be set in the ```[ModelSettings]``` section.  Unless otherwise stated, we assume this type 
of model throughout this README.   
- For Wick-rotated quantum actions, `number_of_quantum_particles` and `number_of_timeslices` should be set in the 
```[ModelSettings]``` section.  Both are `int` types and the function `get_basic_config_data()` in 
[`helper_methods.py`](src/helper_methods.py) sets 
`number_of_particles = number_of_quantum_particles * number_of_timeslices`. 
- `size_of_particle_space` represents the size and dimensions of the spaces on which each particle exists (or each 
quantum particle for Wick-rotated quantum actions).  It is either `None`, a `float` or a Python `list` of `None` or 
`float` values (`None` corresponds to the whole real line).
- `range_of_initial_particle_positions` represents the range of the initial position of each particle (or each quantum 
particle for Wick-rotated quantum actions).  It is either a `float`, a one-dimensional Python `list` of length 
`len(range_of_initial_particle_positions)` and composed of `float` values, or a two-dimensional Python `list` of size 
`(len(range_of_initial_particle_positions), 2)` and composed of `float` values.

The above example represents a two-particle system in which each particle exists on the entire real 
line and has initial position *1.0*, while

```INI
[ModelSettings]
number_of_particles = 4
size_of_particle_space = [1.0, 1.0]
range_of_initial_particle_positions = [[-0.5, 0.5], [-0.5, 0.5]]
```

represents a four-particle system in which each particle exists on the two-dimensional torus (of volume *1.0 x 1.0*) 
and takes an initial position anywhere on that torus.

### Remaining sections

The remaining sections of the configuration file correspond to the different classes chosen for the simulation (i.e. 
the mediator and each of the classes with which it interacts). Each section contains pairs of properties and values. 
Each property corresponds to the name of an argument in the `__init__()` method of the corresponding class, and its 
value provides the argument. Property-value pairs must be provided for all properties that do not have a default value; 
for each property that does have a default value, a property-value pair may be given. Properties and values should be 
given in snake_case; sections should be given in CamelCase. If a value corresponds to the instance of another class, 
then a corresponding section is required. 

### Checkpointing
The super-aLby application also provides checkpointing functionality, allowing simulations to be restarted from some 
previous configuration. The final configuration of the simulated system is saved at 
`path\to\output_directory\configuration_at_checkpoint.npy`, alongside the index of the checkpoint, starting from 0 for 
a system that has not been restarted, saved at `path\to\output_directory\checkpoint_index.txt`. When these files are
present in the output directory, i.e. from a previous simulation of the same config file, the application will load them
and start from that point.

N.B. That unlike some methods of cheeckpointing where the sample is periodically outputted throughout the duration of
the simulation,  in super-aLby the sample is only outputted when the simulation finishes. Therefore, the expected usage
is to run, for example, 10 consecutive simulations with 10,000 samples, as opposed to running a simulation of length
100,000, if simulating for 100,000 samples is likely to fail on the relevant HPC architecture.



### Examples

Building on our example `[Run]` section above, configuration files might be of the form

```INI
[Run]
mediator = some_mediator
number_of_jobs = 1
max_number_of_cpus = 1

[SomeMediator]
potential = some_potential
sampler = some_sampler
kinetic_energy = some_kinetic_energy
...

[SomePotential]
...

[SomeSampler]
...

[SomeKineticEnergy]
...

[ModelSettings]
number_of_particles = 2
size_of_particle_space = None
range_of_initial_particle_positions = 1.0
```

or of the form

```INI
[Run]
mediator = some_mediator
number_of_jobs = 1
max_number_of_cpus = 1

[SomeMediator]
potential = some_potential
sampler = some_sampler
noise_distribution = some_noise_distribution
...

[SomePotential]
...

[SomeSampler]
...

[SomeNoiseDistribution]
...

[ModelSettings]
number_of_particles = 2
size_of_particle_space = None
range_of_initial_particle_positions = 1.0
```

where the ellipsis accounts for further property-value pairs that do not correspond to other classes. The 
first / second example requires the sections `[SomePotential]`, `[SomeSampler]` and `[SomeKineticEnergy]` 
/ `[SomeNoiseDistribution]` because the `[SomeMediator]` section provides [`potential`](src/potential),
[`sampler`](src/sampler) and [`kinetic_energy`](src/kinetic_energy) / [`noise_distribution`](src/noise_distribution) property-value pairs.
The first example must correspond to some form of Hamiltonian Monte Carlo simulation (as it selects a 
[`kinetic_energy`](src/kinetic_energy)) while the second must correspond to a Metropolis Monte Carlo simulation (as it selects a 
[`noise_distribution`](src/noise_distribution)). Note that additional examples are also possible (e.g. one may choose to construct an 
event-chain Monte Carlo simulation).

Some example configuration files are located in the [`src/config_files`](src/config_files) directory. To get a feel for the 
application, run `python run.py 
config_files/convergence_tests/exponential_power_potential_power_equals_4/super_relativistic_kinetic_energy.ini`, 
before running `python sample_analysis/test_convergence.py 
config_files/convergence_tests/exponential_power_potential_power_equals_4/super_relativistic_kinetic_energy.ini` once 
the simulation has finished. 


## *Emergent electrostatics in planar XY spin models* [\[Faulkner2025\]](https://doi.org/10.1088/1367-2630/add7fd)

This details how to make its Ising-related figures.

### Figure 1

Run the script `python sample_analysis/make_ising_spec_heat_and_mag_density_figs.py False`.

### Figure 2

1. Run each configuration file in [`config_files/emergent_electrostatics_ising_figs`](
src/config_files/emergent_electrostatics_ising_figs) via the command 
`python run.py config_files/emergent_electrostatics_ising_figs/4x4_metropolis.ini`, etc.  
2. Once all simulations are complete, run the relevant sample-analysis script via the command 
`python sample_analysis/make_ising_trace_plots.py False`.

### Other figures

For Figures 5-9, 11 and 14-17, go to [xy-type-models](https://github.com/michaelfaulkner/xy-type-models) and follow the 
instructions in the [README](https://github.com/michaelfaulkner/xy-type-models/blob/main/README.md).  We aim to 
eventually integrate [xy-type-models](https://github.com/michaelfaulkner/xy-type-models) into super-aLby.

All other figures are either TikZ-based or some heuristic curve made using matplotlib in a simple Python script.


## *Sampling algorithms in statistical physics* [\[Faulkner2024\]](https://doi.org/10.1214/23-STS893)

This details how to make its Ising-related figures.

To make Figures 2, 9, 10 and 11, first run each configuration file in [`config_files/sampling_algos_ising_figs`](
src/config_files/sampling_algos_ising_figs) via the command `python run.py 
config_files/sampling_algos_ising_figs/4x4_metropolis.ini`, etc.  

Then, once all simulations are complete, run the relevant sample-analysis scripts via the commands 
- `python sample_analysis/make_ising_autocorrelation_figs.py`
- `python sample_analysis/make_ising_spec_heat_and_mag_density_figs.py`
- `python sample_analysis/make_ising_trace_plots.py`

Each script also creates additional figures not presented in the paper.  These may also be useful to the user. 

To make Figure 12, go to [xy-type-models](https://github.com/michaelfaulkner/xy-type-models) and follow the instructions 
in the [README](https://github.com/michaelfaulkner/xy-type-models/blob/main/README.md).  We aim to eventually integrate 
[xy-type-models](https://github.com/michaelfaulkner/xy-type-models) into super-aLby.

All other figures are either TikZ-based or some heuristic curve made using matplotlib in a simple Python script.
