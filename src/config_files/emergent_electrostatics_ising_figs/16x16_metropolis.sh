#!/bin/bash
export TEMPLATE_INI=config_files/emergent_electrostatics_ising_figs/16x16_metropolis.ini
export RUN_WITH_EXECUTABLE=false
export NUM_JOBS=1
export START=1.0
export END=3.666666666666666
export NUM_INCREMENTS=2
export CONFIG_HEADER=MetropolisMediator
export CONFIG_VARIABLE=temperature
export MAX_CPUS=3

exec ./run_spawned_configs.sh
