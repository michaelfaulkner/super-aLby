#!/bin/bash
export TEMPLATE_INI=config_files/optimal_sampling_strategies_figs/fig2/N64/L128_power2_unfactor_sweep.ini
export RUN_WITH_EXECUTABLE=false
export NUM_JOBS=12
export START=0.0
export END=4.0
export NUM_INCREMENTS=40
export CONFIG_HEADER=UnfactorisedHarmonicChainPotential
export CONFIG_VARIABLE=equilibrium_length
export MAX_CPUS=41

exec ./run_spawned_configs.sh
