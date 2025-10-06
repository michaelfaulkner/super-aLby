#!/bin/bash
export TEMPLATE_INI=config_files/harmonic_chain/pressure/sl_squared_temp_100.ini
export NUM_JOBS=7
export START=1.0
export END=100.0
export NUM_INCREMENTS=1
export CONFIG_HEADER=EventChainMediator
export CONFIG_VARIABLE=temperature
export MAX_CPUS=12

exec ./run_spawned_configs.sh
