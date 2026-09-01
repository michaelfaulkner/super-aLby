import numpy as np
import sys
import matplotlib.pyplot as plt


def main(active_particle_index_data_path, timestep, number_of_particles):

    number_of_particles = int(number_of_particles)

    active_particle_index_data = np.load(active_particle_index_data_path).flatten()

    print(len(active_particle_index_data))

    initial_index = active_particle_index_data[0]

    msd = 0.0
    boundaries = 0
    """
    for index in range(len(active_particle_index_data)):
        if index != 0:
            current_active_particle_index  = active_particle_index_data[index]
            boundaries = check_boundary(active_particle_index_data[index-1], current_active_particle_index,
                                        number_of_particles, boundaries)
            if boundaries == 0:
                msd += (current_active_particle_index - initial_index)**2
            else:
                msd += (np.sign(boundaries) * current_active_particle_index +
                        np.abs(boundaries) * number_of_particles - initial_index)**2

    msd = msd / len(active_particle_index_data)
    print(np.sqrt(msd))
    """

    plt.plot(np.arange(len(active_particle_index_data[:10000])), active_particle_index_data[:10000])
    plt.xlabel("event")
    plt.ylabel("particle index")
    plt.savefig("active_particle_index.png")



def check_boundary(index_before, index_after, number_of_particles, boundaries):

    if index_before == number_of_particles - 1 and index_after == 0:
        return boundaries + 1
    elif index_before == 0 and index_after == number_of_particles - 1:
        return boundaries -1
    else:
        return boundaries






if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])