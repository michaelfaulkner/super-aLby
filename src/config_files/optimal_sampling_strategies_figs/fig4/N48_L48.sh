#!/bin/bash
export TEMPLATE_INI=config_files/optimal_sampling_strategies_figs/fig4/N48_L48.ini
export RUN_WITH_EXECUTABLE=false
export NUM_JOBS=10
export START=0.0
export END=3.992
export NUM_INCREMENTS=40
export CONFIG_HEADER=SoftDiskPotential
export CONFIG_VARIABLE=factor_field_prefactor
export MAX_CPUS=41

exec ./run_spawned_configs.sh
