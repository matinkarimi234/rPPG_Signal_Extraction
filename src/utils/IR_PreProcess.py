import pandas as pd
import numpy as np
from BaselineRemoval import BaselineRemoval


def cut_signal(in_sig):
    """
    Cuts the 5 seconds of input signal, starting from the 650th element.

    Parameters:
    - in_sig (numpy.ndarray): Input signal array.

    Returns:
    numpy.ndarray: Limited signal array.
    """

    if len(in_sig) >= 2500:
        in_sig_limited = in_sig[650:]
    else:
        in_sig_limited = in_sig

    return in_sig_limited


def reflect(in_sig):
    """
    Returns the negative of the input signal.

    Parameters:
    - in_sig (numpy.ndarray): Input signal array.

    Returns:
    numpy.ndarray: Reflected signal array.
    """

    return -in_sig


def isoutlier(data):
    """
    Filters outliers from the input data using a 2-sigma filter.

    Parameters:
    - data (list or numpy.ndarray): Input data array.

    Returns:
    list: Filtered data array.
    """

    m = 2
    u = np.mean(data)
    s = np.std(data)
    filtered = [e for e in data if (u - m * s < e < u + m * s)]

    return filtered


def remove_baseline(in_sig):
    """
    Removes baseline from the input signal using the Zhang-Fit method.

    Parameters:
    - in_sig (numpy.ndarray): Input signal array.

    Returns:
    numpy.ndarray: Signal with baseline removed.
    """

    base=BaselineRemoval(in_sig)
    out_sig=base.ZhangFit()

    return out_sig


def clean(in_sig):
    """
    Cleans the input signal by applying various preprocessing steps, including cutting, reflecting,
    handling NaN values, removing baseline, and filtering outliers.

    Parameters:
    - in_sig (numpy.ndarray): Input signal array.

    Returns:
    numpy.ndarray: Cleaned signal array.
    """

    lim_sig= cut_signal(in_sig)
    reflected_sig = reflect(lim_sig)
    Nans = np.isnan(reflected_sig)
    reflected_sig[Nans] = 0
    sig_no_baseline = remove_baseline(reflected_sig)
    sig_no_outlier = isoutlier(sig_no_baseline)
    out_sig= sig_no_outlier

    return out_sig


def preprocessed_datas(num_files: int):
    """
    Reads and preprocesses PPG signals from a specified directory, returning a DataFrame
    containing preprocessed data for multiple signals.

    Returns:
    pandas.DataFrame: DataFrame containing preprocessed PPG signals.
    """
    
    directory = '/Users/aliebrahimi/Documents/PHD/PHD/PPG_Signals/'
    base_filename = 'ir.csv'
    file_addresses = [f'{directory}{i}_{base_filename}' for i in range(1, num_files + 1)]
    dfs = {}
    
    for file_address in file_addresses:
        index = int(file_address.split('/')[-1].split('_')[0])
        df = pd.read_csv(file_address, names=[f'{index}_ir', 'NAN'])
        data_preprocessed= clean(df[f'{index}_ir'])
        dfs[index] = data_preprocessed
    

    min_length = min(len(dfs[i]) for i in range(1, num_files + 1))

    for i in range(1,num_files+1):
        dfs[i] = dfs[i][:min_length]

    p_datas = pd.DataFrame(dfs)

    return p_datas



