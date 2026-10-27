#!/bin/bash
export TEMPLATE_INI=config_files/optimal_sampling_strategies_figs/fig3/3a/L64_power4_naive.ini
export RUN_WITH_EXECUTABLE=false
export NUM_JOBS=10
export START=64.23
export END=64.23
export NUM_INCREMENTS=0
export CONFIG_HEADER=HarmonicChainFactorField
export CONFIG_VARIABLE=prefactor
export MAX_CPUS=2

exec ./run_spawned_configs.sh
