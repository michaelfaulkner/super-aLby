import numpy as np
import os
import importlib
import matplotlib
import matplotlib.pyplot as plt
import sample_getter
import sys
from configparser import NoOptionError
from markov_chain_diagnostics import get_sample_mean_and_error

this_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(this_directory + "/../")
sys.path.insert(0, src_directory)
helper_methods = importlib.import_module("helper_methods")
parsing = importlib.import_module("base.parsing")
strings = importlib.import_module("base.strings")

def main(top_folder):
    #x2_arr = np.zeros(len(os.listdir(top_folder)))
    timestep = 0.01
    fig, ax = plt.subplots(2,2, sharex = True, sharey = True)
    for folder_index, folder in enumerate(os.listdir(top_folder)):
        
        path = os.path.join(top_folder, folder)
       # print(folder_index, path)
        x2_arr = np.zeros(len(os.listdir(path)))
        for index, data_file in enumerate(os.listdir(path)):
            mean_sample = np.load(os.path.join(path, data_file))
            mean_sample_mean = get_sample_mean_and_error(mean_sample)[0] / timestep **2
            x2_arr[index] = mean_sample_mean

        if folder_index == 0:
            ax[0,0].scatter(np.arange(len(x2_arr)), x2_arr, label = f"{folder}")
            print(f"{folder} 00")
        elif folder_index == 1:
            ax[0,1].scatter(np.arange(len(x2_arr)), x2_arr, label = f"{folder}")
            print(f"{folder} 10")
        elif folder_index == 2:
            ax[1,1].scatter(np.arange(len(x2_arr)), x2_arr, label = f"{folder}")
            print(f"{folder} 01")
        else:
            ax[1,0].scatter(np.arange(len(x2_arr)), x2_arr, label = f"{folder}")
            print(f"{folder} 11")
   
    ax[0,0].legend()
    ax[0,1].legend()
    ax[1,0].legend()
    ax[1,1].legend()
    ax[1,0].set_xlabel("iteration no.")
    ax[1,1].set_xlabel("iteration no.")
    ax[0,0].set_ylabel("<x2>")
    ax[1,0].set_ylabel("<x2>")


    
    plt.savefig("x2_variations_constant_sample_num.png")


if __name__ == '__main__':
    main(sys.argv[1])