#!/bin/bash
export TEMPLATE_INI=config_files/harmonic_chain/specific_heat/test_event_chain.ini
export NUM_JOBS=6
export START=1.0
export END=3.0
export NUM_INCREMENTS=29
export CONFIG_HEADER=EventChainMediator
export CONFIG_VARIABLE=temperature
export MAX_CPUS=10

exec ./run_spawned_configs.sh
