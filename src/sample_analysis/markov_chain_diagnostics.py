import math
import numpy as np

def get_sample_mean_and_error(sample):
    """
    Calculate the mean and error of a one-dimensional sample.

    Parameters
    ----------
    sample : numpy.ndarray
        Sample to be analysed.
    
    Returns
    -------
    list
        A list containing the mean and error of the sample [mean, error].
    """
    iat = get_integrated_autocorrelation_time(sample)
    return [np.mean(sample), np.std(sample, ddof=1) * (iat / len(sample)) ** 0.5]


def get_thinned_sample(one_dimensional_sample, thinning_level):
    if len(np.atleast_2d(one_dimensional_sample)) > 1:
        raise Exception("Error: the sample passed to markov_chain_diagnostics.get_thinned_sample() must be one "
                        "(Cartesian) dimensional.")
    sample_indices_to_keep = np.array([i for i in range(len(one_dimensional_sample)) if i % thinning_level == 0])
    return np.take(one_dimensional_sample, sample_indices_to_keep)


def get_cumulative_distribution(one_dimensional_sample):
    if len(np.atleast_2d(one_dimensional_sample)) > 1:
        raise Exception("Error: the sample passed to markov_chain_diagnostics.get_cumulative_distribution() must be "
                        "one (Cartesian) dimensional.")
    """alternative calculation commented out, nb, factor of 1 / 10 (in bins) may not be optimal"""
    """count, bins_count = np.histogram(magnetisation_phase, bins=int(len(one_dimensional_sample) / 10))
    cdf = np.array([bins_count[1:], np.cumsum(count / sum(count))])"""
    bin_values = np.arange(1, len(one_dimensional_sample) + 1) / float(len(one_dimensional_sample))
    ordered_sample = np.sort(one_dimensional_sample)
    return [ordered_sample, bin_values]


def get_autocorrelation(sample):
    """
    Calculate the autocorrelation function of a one-dimensional sample.  

    Parameters
    ----------
    sample : numpy.ndarray
        Sample to be analysed.

    Returns
    -------
    numpy.ndarray
        Autocorrelation function of the sample.
    """
    if len(np.atleast_2d(sample)) > 1:
        raise Exception("Error: the sample passed to markov_chain_diagnostics.get_autocorrelation() must be one "
                        "(Cartesian) dimensional.")
    mean_zero_sample = sample - np.mean(sample)
    full_acf = np.correlate(mean_zero_sample, mean_zero_sample, mode='full')
    """np.correlate() is symmetric about t = 0 when mode='full' - full_acf[full_acf.size // 2:] returns t >= 0 values"""
    acf = full_acf[full_acf.size // 2:]
    acf /= acf[0] # Normalise
    return acf

def get_integrated_autocorrelation_time(sample, cutoff=math.e ** (-2)):
    """
    Calculate the integrated autocorrelation time of a one-dimensional sample.
    
    Parameters
    ----------
    sample : numpy.ndarray
        Sample to be analysed.
    cutoff : float
        Cutoff value for the autocorrelation function. The default value is e^(-2).

    Returns
    -------
    float
        Integrated autocorrelation time.
    """
    autocorrelation_function = get_autocorrelation(sample)
    below_cutoff = np.where(autocorrelation_function < cutoff)[0]
    max_acf_index = below_cutoff[0] - 1
    return 2.0 * np.sum(autocorrelation_function[:max_acf_index]) - 1.0


def get_effective_sample_size(sample):
    """
    Calculate the effective sample size of a one-dimensional sample.
    
    Parameters
    ----------
    sample : numpy.ndarray
        Sample to be analysed.

    Returns
    -------
    float
        Effective sample size.
    """
    if len(np.atleast_2d(sample)) > 1:
        raise Exception("Error: the sample passed to markov_chain_diagnostics.get_effective_sample_size() must be "
                        "one (Cartesian) dimensional.")
    iat = get_integrated_autocorrelation_time(sample, cutoff=math.e ** (-4))
    return len(sample) / iat
