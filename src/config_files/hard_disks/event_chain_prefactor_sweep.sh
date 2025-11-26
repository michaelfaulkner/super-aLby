#!/bin/bash
export TEMPLATE_INI=config_files/hard_disks/event_chain_prefactor_sweep.ini
export NUM_JOBS=5
export START=0.05
export END=1.0
export NUM_INCREMENTS=19
export CONFIG_HEADER=HardDiskFactorField
export CONFIG_VARIABLE=prefactor
export MAX_CPUS=10

exec ./run_spawned_configs.sh
