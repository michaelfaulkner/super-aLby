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
