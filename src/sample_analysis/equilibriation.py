import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib
matplotlib.use('Agg')


def main(positions_data, mean_positions_data, active_particle_index_data, Nt):

    Nt = float(Nt)
  
    positions = np.load(positions_data)
    active_particle_index = np.load(active_particle_index_data)
    print(np.shape(positions))

    plt.scatter(np.arange(len(positions[0, :])), positions[0, :])

    plt.savefig("equlibriate_short_+.png")
    plt.clf()

    plt.scatter(np.arange(len(positions[50, :])), positions[50, :])

    plt.savefig("equlibriate_short_+_1.png")
    plt.clf()

    plt.scatter(np.arange(len(positions[150, :])), positions[150, :])
    
    plt.savefig("equlibriate_short_+_2.png")
    plt.clf()

    plt.scatter(np.arange(len(positions[700, :])), positions[700, :])
    
    plt.savefig("equlibriate_short_+_3.png")
    plt.clf()

    plt.scatter(np.arange(len(positions[800, :])), positions[800, :])
    
    plt.savefig("equlibriate_short_+_4.png")
    plt.clf()

    plt.scatter(np.arange(len(positions[1000, :])), positions[1000, :])
    
    plt.savefig("eequlibriate_short_+_5.png")
    plt.clf()

    
    plt.scatter(np.arange(len(positions[1500, :])), positions[1500, :])
    
    plt.savefig("equlibriate_short_+_6.png")
    plt.clf()

    plt.scatter(np.arange(len(positions[5000, :])), positions[5000, :])
    
    plt.savefig("equlibriate_short_+_7.png")
    plt.clf()


    plt.scatter(np.arange(len(active_particle_index))[:5000], active_particle_index[:5000])
    plt.savefig("active_particle_short_+.png")
    plt.clf()


    mean_positions = np.load(mean_positions_data)

    plt.scatter(np.arange(len(mean_positions[:])), mean_positions[:])
    plt.savefig("mean_positions_equilibriate_short_+.png")
    plt.clf()

    plt.scatter(np.arange(len(mean_positions[:20])), mean_positions[:20])
    plt.savefig("mean_positions_equilibriate_1_short_+.png")
    plt.clf()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])