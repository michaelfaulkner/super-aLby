import numpy as np
import matplotlib.pyplot as plt 

file_path = 'output/convergence_tests/xy_potential/event_chain/temperature_00_checkpoint_00_sample_of_magnetisation_norm.npy'
data = np.load(file_path)

plt.plot(data)
plt.show()