#!/bin/bash
export TEMPLATE_INI=config_files/harmonic_chain/pressure/sl_squared_temp_5.ini
export NUM_JOBS=12
export START=5.0
export END=10.0
export NUM_INCREMENTS=1
export CONFIG_HEADER=EventChainMediator
export CONFIG_VARIABLE=temperature
export MAX_CPUS=12

exec ./run_spawned_configs.sh
