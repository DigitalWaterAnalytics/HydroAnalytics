# python imports

# third party imports
import pandas as pd
import numpy as np


# local imports


def chapman_and_maxwell(data: np.ndarray, a: float = 0.976, initial_baseflow: float = 0.0):
    """
    Baseflow filtration using the Chapman and Maxwell method.

    Chapman, T., 1999. A comparison of algorithms for stream flow recession and baseflow separation.
    Hydrological Processes 13, 701–714.
    https://doi.org/10.1002/(SICI)1099-1085(19990415)13:5<701::AID-HYP774>3.0.CO;2-2

    :param data: numpy array of measured flows
    :param a: recession constant
    :param initial_baseflow: initial baseflow value
    :return: numpy array of baseflow
    """
    length = data.shape[0]
    base_flow = np.zeros((length,))
    base_flow[0] = initial_baseflow
    for i in range(1, length):
        base_flow[i] = (1.0 / (2.0 - a)) * base_flow[i - 1] + ((1.0 - a) / (2.0 - a)) * data[i]
    return base_flow


def eckhardt(data: np.ndarray, a: float = 0.9982, BFI: float = 0.524, initial_baseflow: float = 0.0):
    """

    Baseflow filtration using the Eckhardt method.

    Eckhardt, K.: How to construct recursive digital filters
    for baseflow separation, Hydrol. Process., 19, 507–515,
    doi:10.1002/hyp.5675, 2005.
    :param data:
    :param a:
    :param BFI:
    :param initial_baseflow: initial baseflow value
    :return:
    """
    length = data.shape[0]
    base_flow = np.zeros((length,))
    base_flow[0] = initial_baseflow
    for i in range(1, length):
        base_flow[i] = ((1 - BFI) * a * base_flow[i - 1] + (1.0 - a) * BFI * data[i]) / (1.0 - a * BFI)
    return base_flow


def stewart(data: np.ndarray, k: float = 0.009, f: float = 0.009, initial_baseflow: float = 0.0):
    """
    Stewart, M.K., 2015. Promising new baseflow separation and recession analysis methods applied to
    streamflow at Glendhu Catchment, New Zealand. Hydrology and Earth System Sciences 19, 2587–2603.
    https://doi.org/10.5194/hess-19-2587-2015​

    :param data:
    :param k:
    :param f:
    :param initial_baseflow: initial baseflow value
    :return:
    """
    length = data.shape[0]
    base_flow = np.zeros((length,))
    base_flow[0] = initial_baseflow
    for i in range(1, length):
        if data[i] > base_flow[i - 1] + k:
            flow_diff = data[i] - data[i - 1]
            base_flow[i] = base_flow[i - 1] + k + f * flow_diff
        else:
            base_flow[i] = data[i]
    return base_flow
