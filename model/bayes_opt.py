from bayes_opt import BayesianOptimization, UtilityFunction


class BayesianOptimizationWithCustomAcquisition(BayesianOptimization):
    """:py:class:`BayesianOptimization` that allows using any instance of :py:class:`UtilityFunction`.
    This utility (acquisition) function overrides the one specified in :py:meth:`maximize` method.
    """
    def __init__(self, f, pbounds, utility_function=None, random_state=None, verbose=2):
        super().__init__(f, pbounds, random_state, verbose)

        self._utility_function = utility_function
        if self._utility_function is None:
            self._utility_function = UtilityFunction(kind='ucb', kappa=2.576, xi=0.0)

    def suggest(self, utility_function):
        return super().suggest(self._utility_function)
