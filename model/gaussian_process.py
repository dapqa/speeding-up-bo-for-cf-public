"""This package contains classes implementing custom Bayesian Optimization method from the paper.
"""

import numpy as np

from bayes_opt import Events
from sklearn.gaussian_process.kernels import Matern


class IterationCounter(object):
    """Iteration counter implementation that works as an event listener for :py:class:`BayesianOptimization`."""

    def __init__(self):
        self.counter = 0

    def inc(self):
        self.counter += 1

    def update(self, event, *args, **kwargs):
        if event == Events.OPTIMIZATION_STEP:
            self.inc()


class CustomKernel(Matern):
    """Custom kernel based on a Matern one.
    Modifies covariance of an unseen variable with itself in a such way that dispersion of values predicted by
        a Gaussian Process is increased in the needed area.
    The detailed description is in the paper.
    """

    def __init__(self,
                 length_scale=1.0,
                 length_scale_bounds=(1e-5, 1e5),
                 nu=1.5,
                 iteration_counter=None,
                 sigmoid_scale=1e-3,
                 sigmoid_x_scale=10,
                 sigmoid_x_shift=0,
                 sigmoid_decay_iterations=3
                 ):
        """Constructor.

        :param length_scale: `length_scale` for Matern kernel
        :param length_scale_bounds: `length_scale_bounds` for Matern kernel
        :param nu: `nu` for Matern kernel
        :param iteration_counter: An object that contains `counter` property to get current optimization iteration.
            If None, iteration number is always zero
        :param sigmoid_scale: Vertical scale of a sigmoid
        :param sigmoid_x_scale: Horizontal scale of a sigmoid
        :param sigmoid_x_shift: Horizontal shift of a sigmoid
        :param sigmoid_decay_iterations: Maximal iteration number at which covariance modification is applied
        """
        super().__init__(length_scale, length_scale_bounds, nu)
        self.iteration_counter = iteration_counter
        self.sigmoid_scale = sigmoid_scale
        self.sigmoid_x_scale = sigmoid_x_scale
        self.sigmoid_x_shift = sigmoid_x_shift
        self.sigmoid_decay_iterations = sigmoid_decay_iterations

    def diag(self, X):
        res = super().diag(X)

        iteration_n = 0
        if self.iteration_counter is not None:
            iteration_n = self.iteration_counter.counter

        sigmoid = 1 / (1 + np.exp(-(X[:, 0] + self.sigmoid_x_shift) * self.sigmoid_x_scale))
        sigmoid_multiplier = self.sigmoid_scale * (
                max(self.sigmoid_decay_iterations - iteration_n, 0) / self.sigmoid_decay_iterations)
        mulitplier = 1 + (1 - sigmoid) * sigmoid_multiplier
        res = np.multiply(res, mulitplier)
        return res
