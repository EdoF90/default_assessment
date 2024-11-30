# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.stats import norm, t, multivariate_t


class FeatureSimulator():
    def __init__(self, n_feature, feature_type):
        self.feature_type = feature_type
        self.n_feature = n_feature

    def generate_normalized_features(self, n_data):
        df = None
        if self.feature_type == 'uniform_independent':
            df = pd.DataFrame(
                np.random.uniform(
                    0, 1,
                    size=(n_data, self.n_feature)
                ),
                columns=[chr(n) for n in range(65, 65 + self.n_feature)]
            )
        elif self.n_feature==2 and self.feature_type == '2_squared':
            first_covariate = np.random.uniform(
                0, 1,
                size=n_data
            )
            second_covariate = first_covariate ** 2 + np.random.uniform(
                -0.05, 0.05,
                size=n_data
            )
            df = pd.DataFrame.from_dict(
                {
                    'A': first_covariate,
                    'B': second_covariate,
                }
            )
        elif self.feature_type == 'normal_copula':
            cov = [[0.5, -0.2], [-0.2, 0.5]]
            # generate X
            X = np.random.multivariate_normal(
                (0,0),
                cov,
                n_data
            )
            # get the observation from the copula
            U = norm.cdf(X)
            # compute the observation, since F is uniform in [-1, 1] -> y = F(x) = 0.5 x + 0.5 -> x = 2y -1
            df = pd.DataFrame(
                U,
                columns=[chr(n) for n in range(65, 65 + self.n_feature)]
            )
        elif self.feature_type == 't_copula':
            df = 2
            # define distributions
            multi_rv = multivariate_t([0, 0], [[1, 0.3], [0.3, 1]], df=df)
            uni_rv = t(df=df)
            # sample from multivariate:
            sample = multi_rv.rvs(size = n_data)
            # compute copula:
            U = uni_rv.cdf(sample)
            df = pd.DataFrame(
                U,
                columns=[chr(n) for n in range(65, 65 + self.n_feature)]
            )
        elif self.n_feature==5 and self.feature_type == '5_non_linear':
            linear_covariates = np.random.uniform(
                0, 1,
                size=(n_data, 3)
            )
            df = pd.DataFrame.from_dict(
                {
                    'A': linear_covariates[:, 0],
                    'B': linear_covariates[:, 1],
                    'C': linear_covariates[:, 2],
                    'D': linear_covariates[:, 0] * linear_covariates[:, 1] + np.random.uniform(
                        -0.05, 0.05,
                        size=n_data
                    ),
                    'E': linear_covariates[:, 0] * linear_covariates[:, 2] + np.random.uniform(
                        -0.05, 0.1,
                        size=n_data
                    ),
                }
            )
        elif self.feature_type == '5_difficult':
            linear_covariates = np.random.uniform(
                0, 1,
                size=(n_data, 3)
            )
            df = pd.DataFrame.from_dict(
                {
                    'A': linear_covariates[:, 0],
                    'B': linear_covariates[:, 1],
                    'C': linear_covariates[:, 2],
                    'D': linear_covariates[:, 0] * linear_covariates[:, 1] + np.random.uniform(
                        -0.05, 0.05,
                        size=n_data
                    ),
                    'E': linear_covariates[:, 0] * linear_covariates[:, 2] + np.random.uniform(
                        -0.05, 0.1,
                        size=n_data
                    ),
                }
            )
            
        # if df has not been initialize then there is an error in the setting
        if df is None:
            raise ValueError("FeatureSimulator not valid")
        return df
