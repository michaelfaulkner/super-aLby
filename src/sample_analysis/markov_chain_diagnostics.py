import math
import numpy as np


def get_sample_mean_and_error(sample):
    """
    Calculate the mean and error of an MCMC sample.  The elements of sample must be scalar quantities.

    Parameters
    ----------
    sample : numpy.ndarray
        Sample to be analysed.

    Returns
    -------
    list
        A list containing the mean and error of the sample [mean, error].
    """
    if len(np.atleast_2d(sample)) > 1:
        raise Exception("Error: the sample passed to markov_chain_diagnostics.get_autocorrelation() must be a sample "
                        "of a scalar quantity.")
    iact = get_iact_and_acf(sample)[0]
    return [np.mean(sample), np.std(sample, ddof=1) * (iact / len(sample)) ** 0.5]


def get_thinned_sample(sample, thinning_level):
    """
    Get a thinned sample from sample by keeping every nth element, where n is determined by the thinning level.

    Parameters
    ----------
    sample : numpy.ndarray
        Sample to be thinned.
    thinning_level : int
        The level of thinning, i.e., keep every nth element of the sample.
    """
    sample_indices_to_keep = np.array([i for i in range(len(sample)) if i % thinning_level == 0])
    return np.take(sample, sample_indices_to_keep)


def get_cumulative_distribution(sample):
    """
    Calculate empirical cdf of a sample.  The elements of sample must be scalar quantities.

    Parameters
    ----------
    sample : numpy.ndarray
        Sample to be analysed.
    """
    if len(np.atleast_2d(sample)) > 1:
        raise Exception("Error: the sample passed to markov_chain_diagnostics.get_autocorrelation() must be a sample "
                        "of a scalar quantity.")
    """alternative calculation commented out, nb, factor of 1 / 10 (in bins) may not be optimal"""
    """count, bins_count = np.histogram(magnetisation_phase, bins=int(len(one_dimensional_sample) / 10))
    cdf = np.array([bins_count[1:], np.cumsum(count / sum(count))])"""
    bin_values = np.arange(1, len(sample) + 1) / float(len(sample))
    ordered_sample = np.sort(sample)
    return [ordered_sample, bin_values]


def get_autocorrelation(sample):
    """
    Calculate the autocorrelation function of sample.  The elements of sample must be scalar quantities.

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
        raise Exception("Error: the sample passed to markov_chain_diagnostics.get_autocorrelation() must be a sample "
                        "of a scalar quantity.")
    n = len(sample)
    sample = sample - np.mean(sample)
    f = np.fft.fft(np.hstack([sample, np.zeros(n)]))
    acf = np.fft.ifft(f * np.conj(f))[:n].real
    acf /= np.arange(n, 0, -1)
    if acf[0] < 1.0e-12:
        return acf
    acf /= acf[0]  # Normalise
    return acf


def get_iact(sample, cutoff=math.e ** (-2)):
    """
    Calculate the integrated autocorrelation time and autocorrelation function of sample.  The elements of sample must
        be scalar quantities.

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
    numpy.ndarray
        The autocorrelation function of the sample.
    """
    autocorrelation_function = get_autocorrelation(sample)
    below_cutoff = np.where(autocorrelation_function < cutoff)[0]
    max_acf_index = below_cutoff[0] - 1
    return 2.0 * np.sum(autocorrelation_function[1:max_acf_index]) + 1.0


def get_iact_and_acf(sample, cutoff=math.e ** (-2)):
    """
    Calculate the integrated autocorrelation time and autocorrelation function of sample.  The elements of sample must
        be scalar quantities.

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
    numpy.ndarray
        The autocorrelation function of the sample.
    """
    autocorrelation_function = get_autocorrelation(sample)
    below_cutoff = np.where(autocorrelation_function < cutoff)[0]
    max_acf_index = below_cutoff[0] - 1
    return 2.0 * np.sum(autocorrelation_function[1:max_acf_index]) + 1.0, autocorrelation_function


def get_iact_and_error(sample, cutoff=math.e ** (-2)):
    """
    Calculate the integrated autocorrelation time and autocorrelation function of sample.  The elements of sample must
        be scalar quantities.

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
    numpy.ndarray
        The autocorrelation function of the sample.
    """
    autocorrelation_function = get_autocorrelation(sample)
    below_cutoff = np.where(autocorrelation_function < cutoff)[0]
    max_acf_index = below_cutoff[0] - 1
    iact = 2.0 * np.sum(autocorrelation_function[1:max_acf_index]) + 1.0
    error = get_jackknife_error(sample, round(5*iact), get_iact)
    return iact, error


def get_effective_sample_size(sample):
    """
    Calculate the effective sample size of an MCMC sample.  The elements of sample must be scalar quantities.

    Parameters
    ----------
    sample : numpy.ndarray
        Sample to be analysed.

    Returns
    -------
    float
        Effective sample size.
    """
    iact = get_iact_and_acf(sample, cutoff=math.e ** (-2))[0]
    return len(sample) / iact


def get_jackknife_error(sample, block_size, estimator):
    """
    Calculate the error on a statistical quantity via the jackknife method.

    Parameters
    ----------
    sample : numpy.ndarray
        Sample to be analysed.
    block_size : int
        Size of blocks sample is divided into.
    estimator : function
        Estimator whose error to return.

    Returns
    -------
    float
        Jackknife error.
    """
    sample = list(sample)
    estimate = estimator(sample)
    number_of_blocks = int(len(sample) / block_size)
    if number_of_blocks < 2:
        raise ValueError(f"The jackknife blocksize must be smaller than half the length of the sample. Provided:"
                         f"Sample length={len(sample)}. Block_size={block_size}")
    block_indices = [i * block_size for i in range(number_of_blocks)]
    block_estimators = []
    for i in range(number_of_blocks-1):
        block_sample = np.delete(sample, range(block_indices[i], block_indices[i+1]))
        block_estimators.append(estimator(block_sample))
    jackknife_variance = (number_of_blocks - 1) / number_of_blocks * np.sum((np.array(block_estimators) - estimate)**2)
    return jackknife_variance ** 0.5



