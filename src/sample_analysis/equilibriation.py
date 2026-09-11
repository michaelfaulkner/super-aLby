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

    plt.savefig("test.png")
    plt.clf()

    # plt.scatter(np.arange(len(positions[50, :])), positions[50, :])

    # plt.savefig("equilibriate_1_short_+_dt001.png")
    # plt.clf()

    # plt.scatter(np.arange(len(positions[150, :])), positions[150, :])
    
    # plt.savefig("equilibriate_2_short_+_dt001.png")
    # plt.clf()

    # plt.scatter(np.arange(len(positions[180, :])), positions[180, :])
    
    # plt.savefig("equilibriate_3_short_+_dt001.png")
    # plt.clf()

    # plt.scatter(np.arange(len(positions[200, :])), positions[200, :])
    
    # plt.savefig("equilibriate_4_short_+_dt001.png")
    # plt.clf()

    # plt.scatter(np.arange(len(positions[3500, :])), positions[3500, :])
    
    # plt.savefig("equilibriate_5_short_+_dt001.png")
    # plt.clf()

    
    # plt.scatter(np.arange(len(positions[3800, :])), positions[3800, :])
    
    # plt.savefig("equilibriate_6_short_+_dt001.png")
    # plt.clf()

    # plt.scatter(np.arange(len(positions[5000, :])), positions[5000, :])
    
    # plt.savefig("equilibriate_7_short_+_dt001.png")
    # plt.clf()


    # plt.plot(np.arange(len(active_particle_index))[:500], active_particle_index[:500])
    # plt.savefig("equilibriate_active_particle_+_dt001.png")
    # plt.clf()


    # mean_positions = np.load(mean_positions_data)

    # plt.scatter(np.arange(len(mean_positions[:])), mean_positions[:])
    # plt.savefig("mean_positions_equilibriate_short+_dt001.png")
    # plt.clf()

    # plt.scatter(np.arange(len(mean_positions[:100])), mean_positions[:100])
    # plt.savefig("mean_positions_equilibriate_1_short_+_dt001.png")
    # plt.clf()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])