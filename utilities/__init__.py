# -*- coding: utf-8 -*-
from .computational_functions import FontanaBounds, CVarBounds, beta_bin_quantile, Identity
from .computational_functions import expected_calibration_error, moment_matching, plot_beta_hist
from .data_interface import read_dataset, get_data

pi = lambda a,b: a/(a+b)
rho = lambda a,b: 1/(1+a+b)

__all__ = [
    "FontanaBounds",
    "CVarBounds",
    "expected_calibration_error",
    "beta_bin_quantile",
    "Identity",
    "read_dataset",
    "get_data",
    "pi",
    "rho",
    "moment_matching",
    "plot_beta_hist"
]
