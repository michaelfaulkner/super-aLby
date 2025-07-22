import numpy as np
import matplotlib.pyplot as plt
from sample_analysis.markov_chain_diagnostics import get_cumulative_distribution, get_effective_sample_size

data1 = np.load('output/xy_potential/event_chain/refreshment_ON_vs_O1/O1/8x8/temperature_00_checkpoint_00_sample_of_magnetisation_norm.npy').flatten()
cum1 = get_cumulative_distribution(data1)
data2 = np.load('output/xy_potential/event_chain/refreshment_ON_vs_O1/ON/8x8/temperature_00_checkpoint_00_sample_of_magnetisation_norm.npy').flatten()
cum2 = get_cumulative_distribution(data2)

plt.plot(cum1)
plt.plot(cum2)
plt.savefig('temp.png')
