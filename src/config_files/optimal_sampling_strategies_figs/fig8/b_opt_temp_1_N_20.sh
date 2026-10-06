#!/bin/bash
export TEMPLATE_INI=config_files/optimal_sampling_strategies_figs/fig8/b_opt_temp_1_N_20.ini
export RUN_WITH_EXECUTABLE=false
export NUM_JOBS=12
export START=1.0
export END=1.0
export NUM_INCREMENTS=0
export CONFIG_HEADER=EventChainMediator
export CONFIG_VARIABLE=temperature
export MAX_CPUS=2

exec ./run_spawned_configs.sh
