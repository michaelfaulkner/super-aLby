#!/bin/bash
export TEMPLATE_INI=config_files/sampling_algos_ising_figs/4x4_metropolis.ini
export NUM_JOBS=1
export START=2.22
export END=2.31
export NUM_INCREMENTS=10
export CONFIG_HEADER=MetropolisMediator
export CONFIG_VARIABLE=temperature
export MAX_CPUS=10

exec ./run_spawned_configs.sh
