import numpy as np
import os
import importlib
import sys

this_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(this_directory + "/../")
sys.path.insert(0, src_directory)
helper_methods = importlib.import_module("helper_methods")
parsing = importlib.import_module("base.parsing")
strings = importlib.import_module("base.strings")

def main(file_string, timestep):
    mean_sample = np.load(file_string)
    print(len(mean_sample), timestep)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
