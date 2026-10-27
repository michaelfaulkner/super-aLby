from markov_chain_diagnostics import get_iact, get_iact_and_error
import importlib
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import sample_getter
import sys

# Add the directory that contains the module plotting_functions to sys.path
this_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(this_directory + "/../")
sys.path.insert(0, src_directory)
helper_methods = importlib.import_module("helper_methods")
parsing = importlib.import_module("base.parsing")


def main(config_file_string, sample_name):
    config = parsing.read_config(parsing.parse_options([config_file_string]).config_file)
    (config_file_mediator, potential, _, samplers, sample_directory, _, number_of_equilibration_iterations,
     _, number_of_particles, _) = helper_methods.get_basic_config_data(config_file_string)
    sh_file_string = f"{os.path.splitext(config_file_string)[0]}.sh"
    
    prefactor_values = helper_methods.get_temps_from_bash_file(sh_file_string)
    variable_name = helper_methods.read_variable_from_sh_file(sh_file_string, "CONFIG_VARIABLE")
    
    sample_paths = [
        sorted(glob.glob(os.path.join(sample_directory, f"{variable_name}_{prefactor_index:02d}", "job_*")))
        for prefactor_index in range(len(prefactor_values))
    ]
                     
    iacts = []
    prefactors = []
    for prefactor_index, prefactor_sample in enumerate(sample_paths):
        prefactor_iacts = []
        for sample_path in prefactor_sample:
            try:
                sample = np.load(os.path.join(sample_path, f"checkpoint_00_sample_of_{sample_name}.npy")).flatten()
            except FileNotFoundError:
                continue
            iact = get_iact(sample[number_of_equilibration_iterations:])
            prefactor_iacts.append(iact)
        if len(prefactor_iacts) > 0:
            prefactor_iacts = np.array(prefactor_iacts)
            mean_iact = np.mean(prefactor_iacts)
            iacts.append(mean_iact)
            prefactors.append(prefactor_values[prefactor_index])
            del sample 
            del iact
    
    plt.scatter(prefactors, iacts, color='firebrick', marker='o', linestyle='-', alpha=0.7, linewidth=1.8)

    plt.xlabel("Eq. Length", fontsize=14)
    plt.ylabel("IACT", fontsize=14)

    plt.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.7)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(sample_directory, f"{sample_name}_iact_vs_prefactor.png"))
    plt.show()
    np.save(os.path.join(sample_directory, f"{sample_name}_iact_vs_prefactor.npy"),
            np.vstack([prefactors, iacts]))

    print('Plot saved.')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])