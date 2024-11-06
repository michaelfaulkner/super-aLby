import numpy as np

timesteps = np.arange(0.001,0.5,0.01)

np.savetxt("src/timestep_values_generated.txt", timesteps)