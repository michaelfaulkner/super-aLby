#!/bin/bash
export TEMPLATE_INI=config_files/optimal_sampling_strategies_figs/fig2/N128/L256_power2_unfactor.ini
export RUN_WITH_EXECUTABLE=false
export NUM_JOBS=12
export START=2.0
export END=2.0
export NUM_INCREMENTS=0
export CONFIG_HEADER=UnfactorisedHarmonicChainPotential
export CONFIG_VARIABLE=equilibrium_length
export MAX_CPUS=2

exec ./run_spawned_configs.sh
