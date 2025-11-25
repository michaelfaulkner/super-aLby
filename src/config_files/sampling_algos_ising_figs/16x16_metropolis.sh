#!/bin/bash
export TEMPLATE_INI=config_files/sampling_algos_ising_figs/16x16_metropolis.ini
export NUM_JOBS=28
export START=1.0
export END=3.6
export NUM_INCREMENTS=39
export CONFIG_HEADER=MetropolisMediator
export CONFIG_VARIABLE=temperature
export MAX_CPUS=28

exec ./run_spawned_configs.sh
