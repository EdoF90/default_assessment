# -*- coding: utf-8 -*-
import numpy as np

class ExchangeableLogitSimulator():
    def __init__(self, beta):
        self.beta = beta

    def simulate_defaults(self, normalized_X):
        realization = f(normalized_X)
        p = np.exp(self.beta[0] + np.dot(realization, self.beta[1:])) / (1 + np.exp(self.beta[0] + np.dot(realization, self.beta[1:])))
        return np.random.binomial(1, p)

