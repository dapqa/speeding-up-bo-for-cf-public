"""Utility classes and function for evaluation of Bayesian optimization performance with different kernels
on different datasets.
Most of the functions work in default settings pipeline for the related paper. If needed,
hyperparameter bounds, kernels, and models can be easily replaced.
"""

import os
import time
from typing import List, Optional, Dict, Callable, Any
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from bayes_opt import Events, BayesianOptimization, UtilityFunction
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from model.bayes_opt import BayesianOptimizationWithCustomAcquisition
from model.eigen3_svd import AbstractEigen3SVDModel, Eigen3SVD, Eigen3SVDpp
from model.gaussian_process import IterationCounter, CustomKernel
from util.docker.eigen3_svd import Eigen3SVDService
from util.splits import DatasetSplitFile


class TimeLogger:
    """Event listener for BayesianOptimization instance.
    Logs times for optimization start and each optimization step, which are useful for
    optimization performance estimation.
    """

    def __init__(self):
        self.logged_times: List[float] = []

    def update(self, event, *args, **kwargs):
        if event == Events.OPTIMIZATION_START:
            self.logged_times.clear()
            self.logged_times.append(time.time())
        elif event == Events.OPTIMIZATION_STEP:
            self.logged_times.append(time.time())

    @property
    def times_by_step(self) -> np.ndarray:
        """
        :return: 1-D np.ndarray with i-th cell representing time from optimization start to the finish of i-th step
            (in seconds)
        """
        return np.array(self.logged_times[1:]) - self.logged_times[0]


HyperparameterDenormalizer = Callable[[float, float], Tuple[int, float]]
"""Function that converts normalized float hyperparameters to actual ones with expected types"""

ModelFactory = Callable[[Eigen3SVDService, str, int, float], AbstractEigen3SVDModel]
"""Function that returns an instance of a model,
configured with Eigen3SVDService instance, dataset ID, number of factors and regularization constant
"""

TargetFunction = Callable[[float, float], float]
"""Target function for Bayesian optimization that accepts normalized hyperparameters and returns model's loss"""

KernelFactory = Callable[[IterationCounter], Any]
"""Function that makes a kernel for Gaussian Process. Accepts IterationCounter instance for kernels which
behave differently with respect to current iteration number. 
"""


def eigen3_svd_model_factory(eigen3_svd_service: Eigen3SVDService,
                             dataset_id: str,
                             n_factors: int,
                             reg_weight: float
                             ) -> AbstractEigen3SVDModel:
    """Makes an instance of :py:class:`Eigen3SVD`.
    Implements :py:obj:`ModelFactory` type.

    :param eigen3_svd_service: Docker service for a model
    :param dataset_id: ID of a dataset
    :param n_factors: Number of factors
    :param reg_weight: Regularization constant
    :return: SVD Model
    """
    return Eigen3SVD(
        env_params=eigen3_svd_service.env_params,
        eigen3_svd_service=eigen3_svd_service,
        n_factors=n_factors,
        model_id=f'{dataset_id}-svd-f{n_factors}-regw{reg_weight}',
        lr=0.0075,
        n_epochs=20,
        reg_weight=reg_weight
    )


def eigen3_svdpp_model_factory(eigen3_svd_service: Eigen3SVDService,
                               dataset_id: str,
                               n_factors: int,
                               reg_weight: float
                               ) -> AbstractEigen3SVDModel:
    """Makes an instance of :py:class:`Eigen3SVDpp`.
   Implements :py:obj:`ModelFactory` type.

   :param eigen3_svd_service: Docker service for a model
   :param dataset_id: ID of a dataset
   :param n_factors: Number of factors
   :param reg_weight: Regularization constant
   :return: SVD++ Model
   """
    return Eigen3SVDpp(
        env_params=eigen3_svd_service.env_params,
        eigen3_svd_service=eigen3_svd_service,
        n_factors=n_factors,
        model_id=f'{dataset_id}-svdpp-f{n_factors}-regw{reg_weight}',
        lr=0.0075,
        n_epochs=20,
        reg_weight=reg_weight
    )


def default_hyperparameter_denormalizer(n_factors_norm: float,
                                        reg_weight_norm: float
                                        ) -> Tuple[int, float]:
    """Takes hyperparameter values normalized from default interval to (0; 1] interval,
    and returns actual values for a model.
    Implements :py:obj:`HyperparameterDenormalizer` type.

    :param n_factors_norm: Number of factors [1; 100] -> [0.01; 1]
    :param reg_weight_norm: Regularization constant [0.001; 0.01] -> [0.01; 1]
    :return: Denormalized values of number of factors and regularization constant
    """
    return round(n_factors_norm * 100), reg_weight_norm / 10


def noop_hyperparameter_denormalizer(n_factors_norm: float,
                                     reg_weight_norm: float
                                     ) -> Tuple[int, float]:
    """Does nothing but type conversion.
    Implements :py:obj:`HyperparameterDenormalizer` type.

    :param n_factors_norm: Number of factors
    :param reg_weight_norm: Regularization constant
    :return: Same values of number of factors and regularization constant
    """
    return round(n_factors_norm), reg_weight_norm


def make_target_function(train: DatasetSplitFile,
                         test: DatasetSplitFile,
                         eigen3_svd_service: Eigen3SVDService,
                         model_factory: ModelFactory,
                         inverse_metric: bool = True,
                         hyperparameter_denormalizer: HyperparameterDenormalizer = default_hyperparameter_denormalizer
                         ) -> TargetFunction:
    """Factory function for other functions, which implement :py:obj:`TargetFunction` type.
    Target function that made by this factory does following steps:
        - denormalizes hyperparameters
        - makes a model
        - trains it on a train dataset
        - tests it on a test dataset and returns RMSE of predicted ratings

    :param train: Train dataset file reference
    :param test: Test dataset file reference
    :param eigen3_svd_service: Docker service
    :param model_factory: Factory function that creates models
    :param inverse_metric: If True than target function returns -1 * RMSE
    :param hyperparameter_denormalizer: Function that denormalizes hyperparameters
    :return: Target function that takes normalized hyperparameters and returns model's loss
    """
    def eval_rmse(n_factors_norm: float, reg_weight_norm: float) -> float:
        n_factors, reg_weight = hyperparameter_denormalizer(n_factors_norm, reg_weight_norm)

        model = model_factory(
            eigen3_svd_service,
            train.descriptor.id,
            n_factors,
            reg_weight
        )

        model.fit(train)
        return model.rmse(test)

    if not inverse_metric:
        return eval_rmse
    else:
        def target_function(n_factors_norm: float, reg_weight_norm: float) -> float:
            return -1 * eval_rmse(n_factors_norm, reg_weight_norm)

        return target_function


def make_svd_target_function(train: DatasetSplitFile,
                             test: DatasetSplitFile,
                             eigen3_svd_service: Eigen3SVDService,
                             inverse_metric: bool = True,
                             hyperparameter_denormalizer: HyperparameterDenormalizer = default_hyperparameter_denormalizer
                             ) -> TargetFunction:
    """Same function as :py:func:`make_target_function` but only for SVD models.

    :param train: Train dataset file reference
    :param test: Test dataset file reference
    :param eigen3_svd_service: Docker service
    :param inverse_metric: If True than target function returns -1 * RMSE
    :param hyperparameter_denormalizer: Function that denormalizes hyperparameters
    :return: Target function that takes normalized hyperparameters and returns model's loss
    """
    return make_target_function(
        train=train,
        test=test,
        eigen3_svd_service=eigen3_svd_service,
        model_factory=eigen3_svd_model_factory,
        inverse_metric=inverse_metric,
        hyperparameter_denormalizer=hyperparameter_denormalizer
    )


def make_svdpp_target_function(train: DatasetSplitFile,
                               test: DatasetSplitFile,
                               eigen3_svd_service: Eigen3SVDService,
                               inverse_metric: bool = True,
                               hyperparameter_denormalizer: HyperparameterDenormalizer = default_hyperparameter_denormalizer
                               ) -> TargetFunction:
    """Same function as :py:func:`make_target_function` but only for SVD++ models.

    :param train: Train dataset file reference
    :param test: Test dataset file reference
    :param eigen3_svd_service: Docker service
    :param inverse_metric: If True than target function returns -1 * RMSE
    :param hyperparameter_denormalizer: Function that denormalizes hyperparameters
    :return: Target function that takes normalized hyperparameters and returns model's loss
    """
    return make_target_function(
        train=train,
        test=test,
        eigen3_svd_service=eigen3_svd_service,
        model_factory=eigen3_svdpp_model_factory,
        inverse_metric=inverse_metric,
        hyperparameter_denormalizer=hyperparameter_denormalizer
    )


def make_optimizer(target_function: TargetFunction,
                   kernel_factory: KernelFactory,
                   pbounds: Optional[Dict] = None,
                   utility_function: Optional[UtilityFunction] = None
                   ) -> BayesianOptimization:
    """Makes an instance of :py:class:`BayesianOptimization` that optimizes
    target function using Gaussian Process as surrogate model.
    Gaussian Process if fully defined by its kernel which is made by kernel factory
    function.

    :param target_function: Function to optimize
    :param kernel_factory: Function that makes a kernel for Gaussian Process
    :param pbounds: Bounds for target function parameters.
        If None then default value is used, that is (0.01, 1) for `n_factors_norm`
        and (0.01, 1) for `reg_weight_norm`
    :param utility_function: Utility (Acquisition) function for Bayesian Optimization.
        If None then UCB is used.
    :return:
    """
    if pbounds is None:
        pbounds = {
            'n_factors_norm': (0.01, 1),
            'reg_weight_norm': (0.01, 1)
        }

    optimizer = BayesianOptimizationWithCustomAcquisition(
        f=target_function,
        pbounds=pbounds,
        utility_function=utility_function
    )

    time_logger = TimeLogger()
    optimizer.subscribe(Events.OPTIMIZATION_START, time_logger)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, time_logger)

    iteration_counter = IterationCounter()
    optimizer.subscribe(Events.OPTIMIZATION_STEP, iteration_counter)

    optimizer._gp = GaussianProcessRegressor(
        kernel=kernel_factory(iteration_counter),
        alpha=1e-6,
        normalize_y=False,
        n_restarts_optimizer=0,
        random_state=optimizer._random_state,
    )

    return optimizer


def matern_kernel_factory(nu: float) -> KernelFactory:
    """Function that makes a factory for :py:class:`Matern` kernel.
    Returned factory implements :py:obj:`KernelFactory` type.

    :param nu: nu for Matern kernel
    :return: Kernel factory function
    """
    def res(iteration_counter):
        return Matern(nu=nu)

    return res


def custom_kernel_factory(nu: float,
                          sigmoid_scale: float = 1e-3,
                          sigmoid_decay_iterations: int = 5,
                          ):
    """Function that returns :py:class:`CustomKernel` that based on :py:class:`Matern` one
    but modifies covariance for unseen values (as described in the paper).

    :param nu: nu for CustomKernel
    :param sigmoid_scale: sigmoid_scale for CustomKernel
    :param sigmoid_decay_iterations: sigmoid_decay_iterations for CustomKernel
    :return: Kernel factory function
    """
    def res(iteration_counter):
        return CustomKernel(
            nu=nu,
            iteration_counter=iteration_counter,
            sigmoid_scale=sigmoid_scale,
            sigmoid_decay_iterations=sigmoid_decay_iterations,
            sigmoid_x_shift=0
        )

    return res


def __evaluated_optimizers_dict_to_df(evaluated_optimizers_dict: Dict[str, List[BayesianOptimization]],
                                      n_factors_arg_name: str = 'n_factors_norm',
                                      reg_weight_arg_name: str = 'reg_weight_norm'
                                      ) -> pd.DataFrame:
    """Converts experiment results from raw Python objects to Pandas DataFrame.

    :param evaluated_optimizers_dict: Keys are experiment IDs, values are lists of
        optimizer instances which have completed full optimization cycle
    :param n_factors_arg_name: Argument name for "number of factors" in used parameter bounds.
    :param reg_weight_arg_name: Argument name for "regularization constant" in used parameter bounds.
    :return: Pandas DataFrame indexed by experiment ID ("optimizer"), evaluation number, and iteration number.
        Contains used values of parameters, model RMSE, and time passed from optimization start (in seconds).
    """
    optimizer_names: List[str] = []
    evaluation_indices: List[int] = []
    opt_iter_indices: List[int] = []
    opt_iter_regw: List[float] = []
    opt_iter_n_factors: List[float] = []
    opt_iter_rmse: List[float] = []
    opt_iter_times: List[float] = []

    for optimizer_name, evaluations in evaluated_optimizers_dict.items():
        ev_idx: int = 0
        for optimizer in evaluations:
            time_logger: TimeLogger = next(filter(
                lambda k: type(k) == TimeLogger,
                optimizer._events[Events.OPTIMIZATION_STEP].keys()
            ))

            times_by_step = time_logger.times_by_step

            for i in range(0, len(optimizer.res)):
                optimizer_names.append(optimizer_name)
                evaluation_indices.append(ev_idx)
                opt_iter_indices.append(i)
                opt_iter_n_factors.append(optimizer.res[i]['params'][n_factors_arg_name])
                opt_iter_regw.append(optimizer.res[i]['params'][reg_weight_arg_name])
                opt_iter_rmse.append(-optimizer.res[i]['target'])
                opt_iter_times.append(times_by_step[i])

            ev_idx += 1

    results_df_idx = pd.MultiIndex.from_arrays(
        [optimizer_names, evaluation_indices, opt_iter_indices],
        names=('optimizer', 'evaluation_no', 'evaluation_iter_no')
    )
    results_df: pd.DataFrame = pd.DataFrame(
        index=results_df_idx,
        columns=['n_factors', 'reg_weight', 'rmse', 'time'],
        data=np.array([opt_iter_n_factors, opt_iter_regw, opt_iter_rmse, opt_iter_times]).T
    )

    return results_df


def evaluate_optimizer_or_load_dump(experiment_settings: dict,
                                    optimization_iter_count: int = 10,
                                    optimizer_evaluations_count: int = 10,
                                    optimizer_init_points: int = 2,
                                    optimizer_probes: List = None,
                                    dump_file_name: Optional[str] = None,
                                    ) -> pd.DataFrame:
    """Performs experiments on Bayesian Optimization with given settings.
    First param is a dictionary, where keys are experiment IDs, values are dictionaries
    which set optimizer factories. E.g.::

        'Matern 2.5 + ucb (ip 1)': {
            'optimizer_factory': make_optimizer_factory(
                matern_kernel_factory(nu=2.5),
            )
        }

    where 'optimizer_factory' is a function that takes no arguments and creates an instance of
    :py:class:`BayesianOptimization` or its child.
    Each optimizer is created and evaluated multiple times, using given iteration count,
    random probes count, and predefined probes if they're passed
    (see :py:meth:`BayesianOptimization.probe`).

    Resulting Pandas DataFrame is in format defined by function :py:func:`__evaluated_optimizers_dict_to_df`.

    If name of a dump file is passed, this function checks if this file exists.
    In case of existing dump presence, results are loaded from it and returned without any evaluation.
    In other case evaluation will be fully performed and results will be saved in a new file.

    :param experiment_settings: Evaluation settings (experiment IDs and optimizer factories)
    :param optimization_iter_count: Number of optimization iterations
    :param optimizer_evaluations_count: Number of evaluations of each experiment
    :param optimizer_init_points: Number of initial random probes
    :param optimizer_probes: Predefined initial random probes
    :param dump_file_name: Name of a dump file (relative to project root)
    :return: Pandas DataFrame with evaluation results
    """
    if dump_file_name is not None and os.path.exists(dump_file_name):
        results_df = pd.read_csv(
            dump_file_name,
            index_col=['optimizer', 'evaluation_no', 'evaluation_iter_no']
        )

        print(f'Results are loaded from {dump_file_name}')
        return results_df

    evaluated_optimizers_dict: Dict[str, List[BayesianOptimization]] = dict()
    for experiment_name, experiment_params in experiment_settings.items():
        evaluated_optimizers = []
        for i in range(optimizer_evaluations_count):
            print(f'Evaluating optimizer {experiment_name} #{i + 1}')
            optimizer = experiment_params['optimizer_factory']()

            if optimizer_probes is not None:
                for point in optimizer_probes:
                    optimizer.probe(params=point, lazy=True)

            optimizer.maximize(
                init_points=optimizer_init_points,
                n_iter=optimization_iter_count
            )

            print(optimizer.max)
            evaluated_optimizers.append(optimizer)

        evaluated_optimizers_dict[experiment_name] = evaluated_optimizers

    results_df = __evaluated_optimizers_dict_to_df(evaluated_optimizers_dict)

    if dump_file_name is not None:
        results_df.to_csv(dump_file_name)
        print(f'Results are saved to {dump_file_name}')

    return results_df


def evaluate_all_optimizers_or_load_dump(experiment_settings: dict) -> pd.DataFrame:
    """
    High-level wrapper for :py:func:`evaluate_optimizer_or_load_dump` function.
    Allows performing multiple experiments and saving/loading their results into/from multiple
    dump files (one per experiment).

    All the settings are defined by passed dictionary, e.g.::

        {
            'evaluation_settings': {
                'optimization_iter_count': 10,
                'optimizer_evaluations_count': 10,
                'optimizer_init_points': 1,
                'optimizer_probes': None
            },
            'evaluations': [
                {
                    'dump_file_name': 'bo_ml1m_ip1_m25_ucb_x10_no1.csv',
                    'optimizers': {
                        'Matern 2.5 + ucb (ip 1)': {
                            'optimizer_factory': make_optimizer_factory(
                                matern_kernel_factory(nu=2.5),
                            )
                        }
                    }
                }
            ]
        }

    Parameters in dictionary are equal to :py:func:`evaluate_optimizer_or_load_dump` ones,
    but 'dump_file_name' must be relative to `results` directory (in order to ease the usage in notebooks).

    All the results are concatenated into single Pandas DataFrame.

    :param experiment_settings: Evaluation settings for multiple experiments
    :return: Pandas DataFrame with results of evaluation of all the experiments
    """
    results_list: List[pd.DataFrame] = []
    for evaluation in experiment_settings['evaluations']:
        cur_dump_file_name = None
        if 'dump_file_name' in evaluation:
            cur_dump_file_name = evaluation['dump_file_name']
            cur_dump_file_name = os.path.join('results', cur_dump_file_name)

        results_list.append(evaluate_optimizer_or_load_dump(
            evaluation['optimizers'],
            optimization_iter_count=experiment_settings['evaluation_settings']['optimization_iter_count'],
            optimizer_evaluations_count=experiment_settings['evaluation_settings']['optimizer_evaluations_count'],
            optimizer_init_points=experiment_settings['evaluation_settings']['optimizer_init_points'],
            optimizer_probes=experiment_settings['evaluation_settings']['optimizer_probes'],
            dump_file_name=cur_dump_file_name
        ))

    return pd.concat(results_list)


def mean_confidence_interval(data, confidence=0.95):
    """Computes data mean and bounds of confidence interval.
    This function is taken from https://stackoverflow.com/a/15034143 and authored by https://stackoverflow.com/users/2457899/gcamargo.

    :param data: Array-like numeric data
    :param confidence: Confidence value
    :return: Mean, lower confidence bound, and upper confidence bound
    """
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


def plot_bo_rmse(evaluated_optimizers_df: pd.DataFrame,
                 figsize: Tuple[int, int] = (8, 5),
                 fig=None, ax_time=None, ax_iter=None
                 ) -> None:
    """Plots minimal RMSE values per iteration & per time intervals.

    :param evaluated_optimizers_df: Pandas DataFrame w/ index columns 'optimizer', 'evaluation_no', 'evaluation_iter_no',
        which are name of an experiment, evaluation number, and iteration number within this evaluation
        consequently.
    :param figsize: Optional, matplotlib figure size in inches. Will be used only when a figure is created by this function.
    :param fig: Optional, matplotlib figure to depict the plots.
    :param ax_time: Optional, matplotlib axis to depict 'min RMSE by time' plot.
        If the figure is passed and this axis is not, that plot will not be rendered.
    :param ax_iter: Optional, matplotlib axis to depict 'min RMSE by iteration' plot.
        If the figure is passed and this axis is not, that plot will not be rendered.
    :return: None
    """
    if fig is None:
        fig, (ax_time, ax_iter) = plt.subplots(nrows=2, figsize=figsize)

    # Round-robin plot color, unique per experiment
    color_counter = 0
    # Unique experiment identifiers
    experiment_ids = evaluated_optimizers_df.index.unique(level='optimizer')

    for experiment_id in experiment_ids:
        optimizer_eval_df = evaluated_optimizers_df.loc[experiment_id, :, :]
        evaluation_numbers = optimizer_eval_df.index.unique('evaluation_no')

        # Number of optimization iterations within each evaluation
        # May rarely differ between evaluations since optimization process may stuck before iteration count is reached
        iteration_count = len(optimizer_eval_df.loc[evaluation_numbers[0], :])

        # Numpy 2d arrays, where axis 0 is evaluation number, and axis 1 is iteration number.
        # The first array contains minimal RMSE values per iteration.
        # The second array contains times passed from optimization start.
        min_rmse_by_iter = None
        iter_times = None

        for evaluation_no in evaluation_numbers:
            cur_eval_df = optimizer_eval_df.loc[evaluation_no, :]

            # Making a list of minimal RMSE reached within current evaluation at iteration no. == list idx
            min_rmse_list = cur_eval_df['rmse'].tolist()
            for i in range(len(min_rmse_list)):
                min_rmse_list[i] = min(min_rmse_list[:i + 1])

            times_by_step = cur_eval_df['time'].to_numpy()

            # Since iteration counts may differ from one evaluation to another, if actual number of iterations is less
            # or greater than stated one, we are fixing it
            if len(min_rmse_list) < iteration_count:
                min_rmse_list = min_rmse_list + min_rmse_list[-1:] * (iteration_count - len(min_rmse_list))
            elif len(min_rmse_list) > iteration_count:
                min_rmse_list = min_rmse_list[:iteration_count]

            if min_rmse_by_iter is None:
                min_rmse_by_iter = np.array(min_rmse_list).reshape(1, iteration_count)
                iter_times = times_by_step.reshape(1, iteration_count)
            else:
                min_rmse_by_iter = np.concatenate(
                    [min_rmse_by_iter, np.array(min_rmse_list).reshape(1, iteration_count)], axis=0)
                iter_times = np.concatenate([iter_times, times_by_step.reshape(1, iteration_count)], axis=0)

        # Plotting part

        current_color_code = f'C{color_counter}'
        color_counter += 1

        # min. RMSE by iteration plot
        if ax_iter is not None:
            means = []
            means_plus_h = []
            means_minus_h = []
            for i in range(iteration_count):
                mean, mean_plus_h, mean_minus_h = mean_confidence_interval(min_rmse_by_iter[:, i])
                means.append(mean)
                means_plus_h.append(mean_plus_h)
                means_minus_h.append(mean_minus_h)

            ax_iter.plot(range(1, len(means) + 1), means, color=current_color_code, label=experiment_id)
            ax_iter.fill_between(range(1, len(means) + 1), means_minus_h, means_plus_h, color=current_color_code,
                                 alpha=.1)

            ax_iter.legend()
            ax_iter.set_xlabel('Iteration')
            ax_iter.set_ylabel('RMSE')

        # min. RMSE by time interval plot
        if ax_time is not None:
            # We need to plot mean values per time intervals, but experimental data is naturally no aligned.
            # Since this, we choose a time step and find minimal RMSE values for this time, acquiring per-iteration-like
            # array.
            min_time = np.min(iter_times)
            max_time = np.max(iter_times)
            time_step_s = 0.25  # Time step in seconds

            time_step_count = round((max_time - min_time) / time_step_s) + 1
            min_rmse_by_time_step = np.full((len(evaluation_numbers), time_step_count), np.nan)  # Like min_rmse_by_iter
            for step_idx in range(time_step_count):
                for evaluation_no in range(len(evaluation_numbers)):
                    # A part of min_rmse_by_iter, which contains values filtered by current time step
                    min_rmse_by_iter_filtered = min_rmse_by_iter[
                        evaluation_no,
                        iter_times[evaluation_no, :] <= min_time + step_idx * time_step_s
                    ]

                    if len(min_rmse_by_iter_filtered) > 0:
                        min_rmse_by_time_step[evaluation_no, step_idx] = np.min(min_rmse_by_iter_filtered)

            time_means = []
            time_means_plus_h = []
            time_means_minus_h = []
            for i in range(time_step_count):
                mean, mean_plus_h, mean_minus_h = mean_confidence_interval(min_rmse_by_time_step[:, i])
                time_means.append(mean)
                time_means_plus_h.append(mean_plus_h)
                time_means_minus_h.append(mean_minus_h)

            ax_time.plot(np.linspace(min_time, max_time + 1, len(time_means)), time_means, color=current_color_code,
                         label=experiment_id)
            ax_time.fill_between(np.linspace(min_time, max_time + 1, len(time_means)), time_means_minus_h,
                                 time_means_plus_h, color=current_color_code, alpha=.1)

            ax_time.legend()
            ax_time.set_xlabel('Time (s)')
            ax_time.set_ylabel('RMSE')

    fig.tight_layout()
    fig.show()
