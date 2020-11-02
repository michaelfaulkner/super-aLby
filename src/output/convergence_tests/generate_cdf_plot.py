import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def main():
    matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
    reference_sample = np.loadtxt('output/convergence_tests/gaussian_potential/gaussian_reference_sample.csv', dtype=float, delimiter=',')
    reference_cdf = get_cumulative_distribution(reference_sample)
    sample = np.loadtxt('output/convergence_tests/gaussian_potential/gaussian_kinetic_energy/sample_of_positions.csv', dtype=float, delimiter=',')
    sample_cdf = get_cumulative_distribution(sample[10001:])
    plt.plot(reference_cdf[0], reference_cdf[1], color='r', linewidth=3, linestyle='-', label='reference data')
    plt.plot(sample_cdf[0], sample_cdf[1], color='k', linewidth=2, linestyle='-', label='super-rel-mc data')

    """plt.xlabel(r"Normalized potential ($x$)", fontsize=15, labelpad=10)
    plt.ylabel(r"$\pi \left( \beta \mathbb{E} \left[ U \right] / L^2 < x \right)$", fontsize=15, labelpad=10)
    plt.tick_params(axis='both', which='major', labelsize=14, pad=10)
    legend = plt.legend(loc='lower right', fontsize=10)
    legend.get_frame().set_edgecolor('k')
    legend.get_frame().set_lw(1.5)
    plt.savefig(model_and_sampler_params.open_pdf_file('cumulative_distribution', sample_directory),
                bbox_inches='tight')"""
    plt.show()


def get_cumulative_distribution(input_sample):
    bin_values = np.arange(1, len(input_sample) + 1) / float(len(input_sample))
    ordered_input_sample = np.sort(input_sample)
    return ordered_input_sample, bin_values


if __name__ == '__main__':
    main()
