#!/bin/bash
export TEMPLATE_INI=config_files/optimal_sampling_strategies_figs/fig3/3a/L96_power4.ini
export RUN_WITH_EXECUTABLE=false
export NUM_JOBS=10
export START=216.0
export END=216.0
export NUM_INCREMENTS=0
export CONFIG_HEADER=PolynomialPotential
export CONFIG_VARIABLE=factor_field_prefactor
export MAX_CPUS=2

exec ./run_spawned_configs.sh
