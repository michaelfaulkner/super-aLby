import os 
import re
import sys 
import glob 
import importlib
import numpy as np
import matplotlib.pyplot as plt

this_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(this_directory + "/../")
sys.path.insert(0, src_directory)
parsing = importlib.import_module("base.parsing")
helper_methods = importlib.import_module("helper_methods")

plt.style.use('ggplot')

def main(config_file_string):
    """
    Plot magnetisation temperature sweep and fit for critial exponent beta. 

    Parameters
    ----------
    config_file_string : str 
        Location of the config file.
    """
    config_file_path = parsing.parse_options([config_file_string]).config_file
    config = parsing.read_config(config_file_path)
    (config_file_mediator, potential, samplers, sample_directories, temperatures, number_of_equilibration_iterations,
     _, number_of_particles, _, _, _) = helper_methods.get_basic_config_data(config_file_string)
    
    directory_path = sample_directories[0]

    magnetisations = []

    for temp_index, temperature in enumerate(temperatures):
        file_path = os.path.join(directory_path, f'temperature_{temp_index:02d}_checkpoint_*_sample_of_magnetisation_norm.npy')
        sample_paths = glob.glob(file_path)
        sample_paths = sorted(sample_paths, key=lambda fname: int(re.search(r"checkpoint_(\d{2})", fname).group(1)))

        sample = []
        for sample_path in sample_paths:
            sample.append(np.load(sample_path).flatten())
        sample = np.concatenate(sample)
        sample = sample[number_of_equilibration_iterations:] # Remove burn-in
        
        magnetisation = np.mean(sample)
        magnetisations.append(magnetisation)
    
    temperatures = np.array(temperatures)
    magnetisations = np.array(magnetisations)

    temp_star = helper_methods.get_temp_star(temperatures, magnetisations, number_of_particles)
    temp_KT = 0.893
    temp_c = 4 * temp_star - 3 * temp_KT

    # Plot 1: M vs T
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(temperatures, magnetisations, color='firebrick', alpha=0.6, label=f'{potential} {config_file_mediator}')
    ax.axvline(x=temp_star, alpha=0.6, label=f'T*={temp_star:.2f}', color='k', linestyle='--')
    ax.axvline(x=temp_c, alpha=0.6, label=f'Tc={temp_c:.2f}', color='k', linestyle=':')
          
    critical_temp_range = 0.1
    mask = (temperatures < temp_star + critical_temp_range) & (temperatures > temp_star - critical_temp_range) & (temperatures < temp_c)
    critical_temp_values = temperatures[mask]
    critical_magnetisation_values = magnetisations[mask]
    
    fit_model_params, fit_model_cov, fit_x_values, chi2_reduced = helper_methods.fit_curve(fit_func=lambda x,b,beta:b*x**beta, X=temp_c-critical_temp_values, Y=critical_magnetisation_values, p0=[1.0, 0.23])
    b_fit, beta_fit = fit_model_params
    ax.plot(temp_c - fit_x_values, b_fit * fit_x_values ** beta_fit, 'k-',
    label=f'Non-linear fit (b={b_fit:.3g} β={beta_fit:.3g} χ²ᵣ={chi2_reduced:.1e})')
    
    ax.set_xlabel('T', fontsize=14, color='black')
    ax.set_ylabel('Mean Magnetisation', fontsize=14, color='black')
    ax.set_ylim(0, 1)
    ax.set_title('Mean Magnetisation vs T', fontsize=18, color='black')
    ax.legend(frameon=True, facecolor='white', edgecolor='none', fontsize=10, loc='upper right')
    plt.savefig(os.path.join(directory_path, f'magnetisation_vs_temp.png'))

if __name__ == '__main__':
    main(sys.argv[1])
