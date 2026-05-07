# super-aLby
super-aLby is a Python application that implements various Monte Carlo sampling algorithms for both classical *N*-body 
models in statistical physics (including Wick-rotated quantum actions) and Bayesian probability models.  The Monte Carlo
algorithms include the Metropolis, event-chain, super-relativistic, Hamiltonian, Wolff and Swendsen-Wang Monte Carlo 
algorithms.

For a closely connected discussion of kinetic-energy choice in Hamiltonian/hybrid Monte Carlo, see 
[\[Livingstone2019\]](https://academic.oup.com/biomet/article-abstract/106/2/303/5476364), where we first introduced 
super-relativistic Monte Carlo (though we did not name it).  super-aLby in fact started life as an application for 
Hamiltonian and super-relativistic Monte Carlo (hence the name super-aLby, in reference to Einstein). 
## Contents
1. [Installation](#installation)
2. [Implementation](#implementation)
3. [Configuration files](#configuration-files)
4. [C++ Functionality Using Pybind11](#c-functionality-using-pybind11) 
5. [Running multiple simulations](#running-multiple-simulations)
6. [Checkpointing](#checkpointing)
7. [Published works](#published-works)
    1. [*Emergent electrostatics in planar XY spin models*](#emergent-electrostatics-in-planar-xy-spin-models)
    2. [*Sampling algorithms in statistical physics*](#sampling-algorithms-in-statistical-physics)

## Installation

super-aLby was written using Python 3.8 and C++ but is likely to support any Python version >= 3.6.  It has been tested 
with CPython.

The C++ functionality was introduced to rewrite slow Python functions in C++, but the majority of the application runs 
independently of any C++ code  (see [Implementation](#implementation)).  As such, there are two methods for installing the 
application.

### Plug-and-play installation

To install super-aLby without the C++ functionality, clone this repository.

super-aLby depends on [`numpy`](https://numpy.org).  Some of the sample-analysis code (i.e. scripts contained in the 
[`sample_analysis`](src/sample_analysis) directory) also depends on [`matplotlib`](https://matplotlib.org).  
Plug-and-play installation therefore requires these packages.

### Full installation

To install super-aLby with the C++ functionality, clone this repository then navigate to the top directory and execute 
`./create_env.sh`.  This builds the `super-aLby` executable in the [`src`](src) directory.  The bash script 
[`create_env.sh`](create_env.sh) loads the correct Python environment then runs [`Makefile`](Makefile) which builds the 
executable.  This Make functionality allows developers to rewrite slow Python functions in C++.  This is achieved via 
[C++ Functionality Using Pybind11](#c-functionality-using-pybind11).

This full installation of super-aLby requires up-to-date versions of Python and a C++ compiler.

## Implementation

The user interface of the super-aLby application consists of a configuration file and the [`run.py`](src/run.py) script 
(plug-and-play implementation) or the `super-alby` executable (full implementation).  Note that the `super-alby` 
executable calls the [`run.py`](src/run.py) script, which is located in the [`src`](src) directory. 

The [`run.py`](src/run.py) script and `super-alby` executable expect the path to the configuration file as the first 
positional argument.  Configuration files should be located in the [`config_files`](src/config_files) directory and follow the 
[INI-file format](https://en.wikipedia.org/wiki/INI_file).

To run the super-aLby application, open your terminal, navigate to the [`src`](src) directory and enter `python run.py 
<configuration file>` (plug-and-play implementation) or `./super-alby <configuration file>` (full implementation).  The 
generated sample data will then appear at a location defined in the configuration file (we advise this location to be 
contained within the [`output`](src/output) directory).  Sample analysis can then be performed via scripts within the 
[`sample_analysis`](src/sample_analysis) directory.

Note that the plug-and-play implementation requires that your configuration file does not use any C++ functionality. All
references to plug-and-play implementation will assume that this is the case.

We also provide bash-script functionality for running multiple simulations (possibly in parallel) with the same and/or 
different fixed values of model parameters.  This is described below in the section 
[Running multiple simulations](#running-multiple-simulations).

The super-alby script also takes optional arguments. These are:
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
```

and 

```INI
[ModelSettings]
number_of_particles = 2
size_of_particle_space = None
range_of_initial_particle_positions = 1.0
```

### The mediator

In the above `[Run]` section, `some_mediator` corresponds to the mediator used in the `run.py` file. The mediator 
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
4. [`sampler`](src/sampler) which provides functionality for sampling various system observables and recording event information
in event-chain Monte Carlo (the latter inherit from EventSampler, which inherits from Sampler).

The mediator interacts with (instances of) classes contained in each of these packages, such that the classes never 
interact with each other.  In addition, some [`potential`](src/potential) classes instantiate classes contained in the 
[`linked_lists`](src/linked_lists) package.  This provides linked-list functionality for cell-based evaluation of certain 
multi-particle models defined on the two- or three-dimensional torus.  Below we detail how the configuration file 
chooses the classes that will be used.

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

### Examples

Building on our example `[Run]` section above, configuration files might be of the form

```INI
[Run]
mediator = some_mediator

[SomeMediator]
potential = some_potential
samplers = some_sampler
kinetic_energy = some_kinetic_energy
temperature = 1.0
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

[SomeMediator]
potential = some_potential
samplers = some_sampler, some_other_sampler
noise_distribution = some_noise_distribution
temperature = 2.0
...

[SomePotential]
...

[SomeSampler]
...

[SomeOtherSampler]
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
event-chain Monte Carlo simulation).  In addition, the first / second example sets the temperature to 1.0 / 2.0 (in 
units of the potential energy, which may be dimensionless).

Some example configuration files are located in the [`src/config_files`](src/config_files) directory. To get a feel for the 
application, run `./super-alby
config_files/convergence_tests/exponential_power_potential_power_equals_4/super_relativistic_kinetic_energy.ini`, 
before running `python sample_analysis/test_convergence.py 
config_files/convergence_tests/exponential_power_potential_power_equals_4/super_relativistic_kinetic_energy.ini` once 
the simulation has finished.

### Output directory

Each sample is saved in the output directory defined (by the `output_directory` property) in the section corresponding 
to the relevant sampler.  We advise that `output_directory` mirrors the location of the configuration file such that 
`output_directory = "output/remaining_path/config_file_name"` for a configuration file with path 
`config_files/remaining_path/config_file_name.ini`.  This stores samples within the [`output`](src/output) directory 
but is not a requirement.

## C++ Functionality Using Pybind11
We provide functionality to call some functions using C++.  This has been used to accelerate the slowest functions 
(of certain models) used by `EventChainMediator` and `MetropolisMediator`, n.b. this functionality is currently only 
available for `QuantumHarmonicOscillatorPotential`.

The C++ functions are 'bound' using [`pybind11`](https://pybind11.readthedocs.io/en/stable/index.html), allowing them to be called from Python.  This requires the 
bindings to be [built](https://pybind11.readthedocs.io/en/stable/compiling.html), either using Make (see [Installation](#installation)) or manually (see 
[Building manually](#building-manually)).

Note that this is the reason for building the executable `super-aLby` (see [Installation](#installation)).

### Structure of C++ code and bindings
For each `Potential` class for which C++ functionality is provided, the corresponding C++ code is in the 
`src/potential/cpp_{potential_name}` directory.  For example, for `QuantumHarmonicOscillator`, it is in 
`src/potential/cpp_quantum_harmonic_oscillator`.  The directory contains: 
1. `cpp_{potential_name}.cpp` and `cpp_{potential_name}.hpp`, which provide the C++ versions of the required functions.
2. `bindings.cpp`, which provides the information that `pybind11` requires to compile the functions into callable Python.
3. `__init__.py`, which tells Python that the directory (when built) contains a Python module. 

### Building manually
It is possible (but not preferred) to build manually.  On Linux and for `QuantumHarmonicOscillator`, users should 
activate a valid Python environment then navigate to the top directory and run:  
 
`$ cd src`
`$ pip install pybind11`
`$ cd potential/cpp_quantum_harmonic_oscillator`
`$ c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) bindings.cpp -o cpp_qho.so`  

(N.B. this builds the `cpp_quantum_harmonic_oscillator` directory; other directories must be built separately.)

## Running multiple simulations

The files [`run_spawned_configs.sh`](src/run_spawned_configs.sh) and [`spawn_configs.py`](src/spawn_configs.py) provide functionality for 
running multiple simulations (possibly in parallel) with the same and/or different fixed values of model parameters 
such as the temperature.  To achieve this, the user must create a bash file of the following form which accompanies a 
corresponding configuration file:

```
#!/bin/bash
export TEMPLATE_INI=config_files/convergence_tests/ising_potential/metropolis.ini
export NUM_JOBS=1
export START=1.2
export END=3.0
export NUM_INCREMENTS=1
export CONFIG_HEADER=MetropolisMediator
export CONFIG_VARIABLE=temperature
export MAX_CPUS=2

exec ./run_spawned_configs.sh
```

This example is [config_files/convergence_tests/ising_potential/metropolis.sh](config_files/convergence_tests/ising_potential/metropolis.sh) 
and accompanies [config_files/convergence_tests/ising_potential/metropolis.ini](config_files/convergence_tests/ising_potential/metropolis.ini),
as set by `TEMPLATE_INI`.  The files are located in the same directory and have mirrored names.  We suggest applying 
this convention to all configuration-bash file pairs.

### Bash-file parameters

1. `TEMPLATE_INI` sets the base configuration file from which the bash file constructs the different simulations.
2. `CONFIG_VARIABLE` sets the parameter (located in the `CONFIG_HEADER` section of `TEMPLATE_INI`) over which the bash file iterates independent simulations.
3. `START` should be equal to the value of `CONFIG_VARIABLE` in `TEMPLATE_INI`.
4. `END` should be equal to the desired final value of `CONFIG_VARIABLE`.
5. `NUM_INCREMENTS` sets the number of equally spaced values of `CONFIG_VARIABLE` between and including `START` and 
`END`.  It must be an integer greater than or equal to zero. In the above example, `CONFIG_VARIABLE=temperature`, 
`START=1.2`, `END=3.0` and `NUM_INCREMENTS=1`, which means that the bash file will run independent simulations with the 
value of `temperature` set to 1.2 and 3.0. If `NUM_INCREMENTS` had been set to 2, then the bash file would run 
independent simulations with the value of `temperature` set to 1.2, 2.1 and 3.0.
6. `NUM_JOBS` sets the number of independent simulations at each fixed set of model parameters.  It must be an 
integer greater than or equal to one.  In the above example, `NUM_JOBS=1`, which means that the bash file will run a 
single simulation at each `CONFIG_VARIABLE` increment. The user may choose to run multiple simulations with fixed model 
parameters.  In this case, set `NUM_INCREMENTS=0` and ensure that `START` and `END` are both equal to the value of 
`CONFIG_VARIABLE` in `TEMPLATE_INI`.  The bash file will then run `NUM_JOBS` simulations with the same value of 
`CONFIG_VARIABLE`.
7. Independent simulations may also be run in parallel.  `MAX_CPUS` sets the maximum number of CPUs that may be accessed 
in parallel. In this example, `MAX_CPUS=2`, which means that the bash file will run the simulations at both 
`CONFIG_VARIABLE` increments in parallel (assuming two CPUs are indeed available). `MAX_CPUS` should be chosen to 
avoid overloading personal machines.

### Running the bash file

To run the super-aLby application using this bash functionality, open your terminal, navigate to the [`src`](src) 
directory and enter `./<bash file>`.  This will then run [`run_spawned_configs.sh`](src/run_spawned_configs.sh) which calls 
[`spawn_configs.py`](src/spawn_configs.py).  The latter spawns new configuration files based on the values chosen in 
the bash file.   The remainder of the [`run_spawned_configs.sh`](src/run_spawned_configs.sh) script then runs super-aLby using each of the 
spawned configuration files (possibly in parallel) before deleting the spawned configuration files.  The user therefore 
never has to run [`run_spawned_configs.sh`](src/run_spawned_configs.sh) or [`spawn_configs.py`](src/spawn_configs.py).  

The generated sample data will then appear in `output_directory` (as defined in the corresponding configuration file) 
appended by `/CONFIG_VARIABLE_NM/job_PQ` (where `NM` and `PQ` correspond to the `CONFIG_VARIABLE` iteration number and 
independent-simulation number, respectively).


## Checkpointing
super-aLby also provides checkpointing functionality, allowing simulations to be restarted from the final configuration 
of a previous simulation.  To allow for this, the final configuration generated by some simulation is saved at 
`output_directory/configuration_at_checkpoint.npy` and the index of the corresponding checkpoint is saved at 
`output_directory/checkpoint_index.txt`.  The checkpoint index is set to zero for a simulation that has not been 
restarted and increases by one each time it is restarted.  If these files are present in `output_directory` (i.e. due 
to a previous simulation of the same configuration file) the application will load the checkpoint configuration and 
start from that state.

N.B. Unlike typical checkpointing methods in which the configuration is periodically outputted during the simulation, 
this method outputs the configuration only when the current simulation has finished.  Supposing the target is 100,000 samples but the simulation is likely to timeout before this is achieved, the user might then request 10,000 samples 
in the configuration file and run the simulation ten times.

## Published works

### *Emergent electrostatics in planar XY spin models*
From [\[Faulkner2025\]](https://doi.org/10.1088/1367-2630/add7fd) 

This details how to make its Ising-related figures.

#### Figure 1

Run the script `python sample_analysis/make_ising_spec_heat_and_mag_density_figs.py False`.

#### Figure 2

1. Run each configuration file in [`config_files/emergent_electrostatics_ising_figs`](
src/config_files/emergent_electrostatics_ising_figs) via the command 
`./super-alby config_files/emergent_electrostatics_ising_figs/4x4_metropolis.ini`, etc.  
2. Once all simulations are complete, run the relevant sample-analysis script via the command 
`python sample_analysis/make_ising_trace_plots.py False`.

#### Other figures

For Figures 5-9, 11 and 14-17, go to [xy-type-models](https://github.com/michaelfaulkner/xy-type-models) and follow the 
instructions in the [README](https://github.com/michaelfaulkner/xy-type-models/blob/main/README.md).  We aim to 
eventually integrate [xy-type-models](https://github.com/michaelfaulkner/xy-type-models) into super-aLby.

All other figures are either TikZ-based or some heuristic curve made using matplotlib in a simple Python script.


### *Sampling algorithms in statistical physics* 
From [\[Faulkner2024\]](https://doi.org/10.1214/23-STS893) 

This details how to make its Ising-related figures.

To make Figures 2, 9, 10 and 11, first run each configuration file in [`config_files/sampling_algos_ising_figs`](
src/config_files/sampling_algos_ising_figs) via the command `./super-alby
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
