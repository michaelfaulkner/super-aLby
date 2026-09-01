import numpy as np
import matplotlib.pyplot as plt
import sys

def main(positions_data, mean_positions_data):
    try:
        positions = np.load(positions_data)
        print(np.shape(positions))

        plt.scatter(np.arange(len(positions[0, :])), positions[0, :])

        plt.savefig("equilibriate.png")
        plt.clf()

        plt.scatter(np.arange(len(positions[8000, :])), positions[8000, :])

        plt.savefig("equilibriate_1.png")
        plt.clf()

        plt.scatter(np.arange(len(positions[9000, :])), positions[9000, :])
        
        plt.savefig("equilibriate_2.png")
        plt.clf()

        plt.scatter(np.arange(len(positions[9000, :])), positions[9000, :])
        
        plt.savefig("equilibriate_3.png")
        plt.clf()
    except:
        pass

    mean_positions = np.load(mean_positions_data)

    plt.scatter(np.arange(len(mean_positions[:])), mean_positions[:])
    plt.savefig("mean_positions_equilibriate.png")
    plt.clf()

    plt.scatter(np.arange(len(mean_positions[:5000])), mean_positions[:5000])
    plt.savefig("mean_positions_equilibriate_1.png")
    plt.clf()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])