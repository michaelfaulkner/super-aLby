import numpy as np
import matplotlib.pyplot as plt

range_of_initial_particle_positions = [-1,1]
number_of_particles = 10
positions = np.array([np.atleast_1d(np.random.uniform(*range_of_initial_particle_positions))
                                for _ in range(number_of_particles)])

timestep = 1.0
dimensionless_positions = positions/timestep

def get_gradient_at_index(positions, index):
    """
    Returns the action gradient at a given index

    Parameters
    ----------
    positions : numpy.ndarray
        A one-dimensional numpy array of size (number_of_particles), indexed by time step; each element
        is a float and represents the position of the worldline at that time step.
    index : int
        The time index, i, of the position being considered.
    Returns
    -------
    float
        The dimensionless action gradient."""
    _dimensionless_omega = 0.1
    _dimensionless_m = _dimensionless_omega
    
    if index == 0:
        return _dimensionless_m * ((2 + _dimensionless_omega**2) * positions[index]
                                        - positions[index+1] - positions[-1])
    elif index == (len(positions) - 1):
        return _dimensionless_m * ((2 + _dimensionless_omega**2) * positions[index]
                                        - positions[0] - positions[index-1])
    else:
        return _dimensionless_m * ((2 + _dimensionless_omega**2) * positions[index]
                                        - positions[index+1] - positions[index-1])
    

def get_action_at_index(dimensionless_position_at_index, dimensionless_position_at_next_index):
        """
        Returns the contribution to the dimensionless action from a given index and position.
        Parameters
        ----------
        index : int
            The time index, i, of the position being considered.
        position_at_index : float
            The position of the particle at that index.
        position_at_next_index : float
            The position of the particle at index+1.
        Returns
        -------
        float
            The contribution to the dimensionless action at the given index.
        """
        _dimensionless_omega = 0.1
        _dimensionless_m = _dimensionless_omega

        return (0.5 * _dimensionless_m * (dimensionless_position_at_next_index - dimensionless_position_at_index)**2 +
                0.5 * _dimensionless_m * _dimensionless_omega**2 * dimensionless_position_at_index**2)   

a = []
a_minus_1 = []
a_plus_1 = []
a_increase = np.zeros(8)

v=1
active_particle_index = 5
initial_gradient_a = -v*get_gradient_at_index(dimensionless_positions, 5)
initial_gradient_a_minus_1 = -v*get_gradient_at_index(dimensionless_positions, 5-1)
initial_gradient_a_plus_1 = -v*get_gradient_at_index(dimensionless_positions, 5+1)
a.append(initial_gradient_a)
a_minus_1.append(initial_gradient_a_minus_1)
a_plus_1.append(initial_gradient_a_plus_1)
a_increase[0]= dimensionless_positions[active_particle_index]


print(f"initial gradients: a-1={initial_gradient_a_minus_1}, a={initial_gradient_a}, a+1={initial_gradient_a_plus_1}")
print(dimensionless_positions[5])
dimensionless_positions[5] += 2.5 
print(dimensionless_positions[5])
next_gradient_a = -v*get_gradient_at_index(dimensionless_positions, 5 )
next_gradient_a_minus_1 = -v*get_gradient_at_index(dimensionless_positions, 5-1)
next_gradient_a_plus_1 = -v*get_gradient_at_index(dimensionless_positions, 5+1)
print(f"next gradients: a-1={next_gradient_a_minus_1}, a={next_gradient_a}, a+1={next_gradient_a_plus_1}")
a.append(next_gradient_a)
a_minus_1.append(next_gradient_a_minus_1)
a_plus_1.append(next_gradient_a_plus_1)
a_increase[1]= dimensionless_positions[active_particle_index]

dimensionless_positions[5] += 2.5 
next_gradient_a = -v*get_gradient_at_index(dimensionless_positions, 5 )
next_gradient_a_minus_1 = -v*get_gradient_at_index(dimensionless_positions, 5-1)
next_gradient_a_plus_1 = -v*get_gradient_at_index(dimensionless_positions, 5+1)
print(f"next gradients: a-1={next_gradient_a_minus_1}, a={next_gradient_a}, a+1={next_gradient_a_plus_1}")
a.append(next_gradient_a)
a_minus_1.append(next_gradient_a_minus_1)
a_plus_1.append(next_gradient_a_plus_1)
a_increase[2]= dimensionless_positions[active_particle_index]

dimensionless_positions[5] += 2.5 
next_gradient_a = -v*get_gradient_at_index(dimensionless_positions, 5 )
next_gradient_a_minus_1 = -v*get_gradient_at_index(dimensionless_positions, 5-1)
next_gradient_a_plus_1 = -v*get_gradient_at_index(dimensionless_positions, 5+1)
print(f"next gradients: a-1={next_gradient_a_minus_1}, a={next_gradient_a}, a+1={next_gradient_a_plus_1}")
a.append(next_gradient_a)
a_minus_1.append(next_gradient_a_minus_1)
a_plus_1.append(next_gradient_a_plus_1)
a_increase[3]= dimensionless_positions[active_particle_index]
v=-1
dimensionless_positions[5] -= 1.0
next_gradient_a = -v*get_gradient_at_index(dimensionless_positions, 5 )
next_gradient_a_minus_1 = -v*get_gradient_at_index(dimensionless_positions, 5-1)
next_gradient_a_plus_1 = -v*get_gradient_at_index(dimensionless_positions, 5+1)
print(f"next gradients: a-1={next_gradient_a_minus_1}, a={next_gradient_a}, a+1={next_gradient_a_plus_1}")
a.append(next_gradient_a)
a_minus_1.append(next_gradient_a_minus_1)
a_plus_1.append(next_gradient_a_plus_1)
a_increase[4]= dimensionless_positions[active_particle_index]

dimensionless_positions[5] -= 1.0
next_gradient_a = -v*get_gradient_at_index(dimensionless_positions, 5 )
next_gradient_a_minus_1 = -v*get_gradient_at_index(dimensionless_positions, 5-1)
next_gradient_a_plus_1 = -v*get_gradient_at_index(dimensionless_positions, 5+1)
print(f"next gradients: a-1={next_gradient_a_minus_1}, a={next_gradient_a}, a+1={next_gradient_a_plus_1}")
a.append(next_gradient_a)
a_minus_1.append(next_gradient_a_minus_1)
a_plus_1.append(next_gradient_a_plus_1)
a_increase[5]= dimensionless_positions[active_particle_index]

dimensionless_positions[5] -= 1.0
next_gradient_a = -v*get_gradient_at_index(dimensionless_positions, 5 )
next_gradient_a_minus_1 = -v*get_gradient_at_index(dimensionless_positions, 5-1)
next_gradient_a_plus_1 = -v*get_gradient_at_index(dimensionless_positions, 5+1)
print(f"next gradients: a-1={next_gradient_a_minus_1}, a={next_gradient_a}, a+1={next_gradient_a_plus_1}")
a.append(next_gradient_a)
a_minus_1.append(next_gradient_a_minus_1)
a_plus_1.append(next_gradient_a_plus_1)
a_increase[6]= dimensionless_positions[active_particle_index]

dimensionless_positions[5] -= 1.0
next_gradient_a = -v*get_gradient_at_index(dimensionless_positions, 5 )
next_gradient_a_minus_1 = -v*get_gradient_at_index(dimensionless_positions, 5-1)
next_gradient_a_plus_1 = -v*get_gradient_at_index(dimensionless_positions, 5+1)
print(f"next gradients: a-1={next_gradient_a_minus_1}, a={next_gradient_a}, a+1={next_gradient_a_plus_1}")
a.append(next_gradient_a)
a_minus_1.append(next_gradient_a_minus_1)
a_plus_1.append(next_gradient_a_plus_1)
a_increase[7]= dimensionless_positions[active_particle_index]

print(a_increase)
plt.scatter(a_increase,a, label="a")
plt.scatter(a_increase,a_minus_1, label="a-1")
plt.scatter(a_increase,a_plus_1, label="a+1")
plt.legend()
plt.savefig("fig.png")