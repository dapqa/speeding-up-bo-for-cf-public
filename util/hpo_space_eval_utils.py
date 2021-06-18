"""Utility classes and function for evaluation of SVD/SVD++ models performance with different numbers of factors
and regularization constants.
"""

import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import FigureBase

from model.eigen3_svd import Eigen3SVD
from util.bo_eval_utils import noop_hyperparameter_denormalizer, make_svd_target_function, make_svdpp_target_function
from util.datasets import DatasetDescriptor
from util.docker.eigen3_svd import Eigen3SVDService
from util.splits import split_for_eigen3_svd


def read_existing_results(existing_results_files: List[str],
                          index_columns: List[str],
                          result_column: str
                          ) -> pd.DataFrame:
    """Reads existing HPO space evaluation results and concatenates them into a single Pandas DataFrame.
    It is assumed that there are a plenty of index columns, which are hyperparameter values,
    and a single result column containing model's loss.

    :param existing_results_files: List of names of existing result files (relative to `results` directory)
    :param index_columns: List of names of index column, which are hyperparameter names
    :param result_column: Name of a resulting column
    :return: Pandas DataFrame with all the results which have been read
    """
    res: pd.DataFrame = pd.DataFrame(columns=index_columns + [result_column])

    for results_file in existing_results_files:
        file_content: pd.DataFrame = pd.read_csv(os.path.join('results', results_file))
        file_content.columns = index_columns + [f'{result_column}_{results_file}']
        res = pd.merge(res, file_content, on=index_columns, how='outer')

    res.set_index(index_columns, inplace=True)
    res[result_column] = res.bfill(axis=1).iloc[:, 0]
    res.drop(columns=[f'{result_column}_{results_file}' for results_file in existing_results_files],
             inplace=True)

    return res


def __eval_hpo_space(eval_params_map: dict,
                     existing_results_files_map: dict,
                     dataset_descriptor: DatasetDescriptor,
                     model_type: str,
                     eigen3_svd_service: Eigen3SVDService
                     ) -> pd.DataFrame:
    """Common function for training and evaluating a SVD/SVD++ model on its hyperparameter
    configuration space (which is built by production of 'number of factors' and
    'regularization constant' values).

    Actual non-normalized values of hyperparameters are passed via `eval_params_map` argument,
    that maps dataset IDs to hyperparameter bounds, e.g.::

        {
            'dataset1': {
                'n_factors_list': [(i + 1) * 10 for i in range(10)],
                'reg_weight_list': [round(i * 0.01, 2) for i in range(10)]
            }
        }

    This function creates a model with default parameters (except passed ones) depending on
    `model_type` value, splits dataset defined by its descriptor (80/20 train/test),
    trains a model and evaluates its RMSE.
    Final results are saved into a CSV file in the `results` directory. Results file name
    is auto-generated, and if file with the same name is exists, it will be overwritten.

    Since there could be some existing results, according file names can be passed as
    `existing_results_files_map` argument, e.g.::

        {
            'dataset1': [
                'd1-svd-f10-100-regw0.0-0.09.csv'
            ]
        }

    If model's RMSE for particular hyperparameter set is presented in one of them,
    evaluation will not be performed.

    :param eval_params_map: Dictionary that sets hyperparameter values to build an HPO space
    :param existing_results_files_map: Dictionary with names of existing results files for datasets
        (to take RMSE from them if possible)
    :param dataset_descriptor: Descriptor of a dataset
    :param model_type: 'svd' or 'svdpp' (for SVD and SVD++ respectively)
    :param eigen3_svd_service: Docker service
    :return: Pandas DataFrame indexed by hyperparameter values and containing RMSE for them
    """
    if model_type not in ['svd', 'svdpp']:
        raise ValueError('Wrong model type. Either "svd" or "svdpp" is expected')

    n_factors_list = eval_params_map[dataset_descriptor.id]['n_factors_list']
    regw_list = eval_params_map[dataset_descriptor.id]['reg_weight_list']

    existing_results_files = existing_results_files_map[dataset_descriptor.id]

    _ = Eigen3SVD.compile(
        factor_counts_list=n_factors_list,
        eigen3_svd_service=eigen3_svd_service,
        recompile_if_exists=False,
        verbose=True
    )
    print('Pre-compilation finished')

    train, test = split_for_eigen3_svd(
        dataset_descriptor=dataset_descriptor,
        env_params=eigen3_svd_service.env_params,
        train_file_name=f'{dataset_descriptor.id}-train-80.csv',
        test_file_name=f'{dataset_descriptor.id}-test-20.csv',
        test_size=0.2
    )
    print('Splitting finished')

    target_function_factory = make_svd_target_function if model_type == 'svd' else make_svdpp_target_function
    target_function = target_function_factory(
        train=train,
        test=test,
        eigen3_svd_service=eigen3_svd_service,
        inverse_metric=False,
        hyperparameter_denormalizer=noop_hyperparameter_denormalizer
    )

    rmse = pd.DataFrame(
        columns=['rmse'],
        index=pd.MultiIndex.from_product([n_factors_list, regw_list], names=['n_factors', 'reg_weight'])
    )

    existing_rmse: pd.DataFrame = read_existing_results(
        existing_results_files,
        index_columns=['n_factors', 'reg_weight'],
        result_column='rmse'
    )

    for n_factors in n_factors_list:
        for regw in regw_list:
            try:
                if existing_rmse is not None:
                    rmse.loc[n_factors, regw] = existing_rmse.loc[n_factors, regw]
                    continue
            except KeyError:
                pass

            rmse.loc[n_factors, regw] = target_function(n_factors, regw)
        print(f'Models with n_factors={n_factors} are evaluated')

    result_file_name = f'{dataset_descriptor.id}-{model_type}' \
                       f'-f{min(n_factors_list)}-{max(n_factors_list)}' \
                       f'-regw{min(regw_list)}-{max(regw_list)}' \
                       f'.csv'

    rmse.reset_index().to_csv(
        os.path.join('results', result_file_name),
        index=False
    )

    return rmse


def eval_svd_hpo_space(eval_params_map,
                       existing_results_files_map,
                       dataset_descriptor: DatasetDescriptor,
                       eigen3_svd_service: Eigen3SVDService
                       ) -> pd.DataFrame:
    """Wrapper for :py:func:`__eval_hpo_space` that assumes the usage of a SVD model.

    :param eval_params_map: Dictionary that sets hyperparameter values to build an HPO space
    :param existing_results_files_map: Dictionary with names of existing results files for datasets
        (to take RMSE from them if possible)
    :param dataset_descriptor: Descriptor of a dataset
    :param eigen3_svd_service: Docker service
    :return: Pandas DataFrame indexed by hyperparameter values and containing RMSE for them
    """
    return __eval_hpo_space(eval_params_map=eval_params_map,
                            existing_results_files_map=existing_results_files_map,
                            dataset_descriptor=dataset_descriptor,
                            eigen3_svd_service=eigen3_svd_service,
                            model_type='svd')


def eval_svdpp_hpo_space(eval_params_map,
                         existing_results_files_map,
                         dataset_descriptor: DatasetDescriptor,
                         eigen3_svd_service: Eigen3SVDService
                         ) -> pd.DataFrame:
    """Wrapper for :py:func:`__eval_hpo_space` that assumes the usage of a SVD++ model.

    :param eval_params_map: Dictionary that sets hyperparameter values to build an HPO space
    :param existing_results_files_map: Dictionary with names of existing results files for datasets
        (to take RMSE from them if possible)
    :param dataset_descriptor: Descriptor of a dataset
    :param eigen3_svd_service: Docker service
    :return: Pandas DataFrame indexed by hyperparameter values and containing RMSE for them
    """
    return __eval_hpo_space(eval_params_map=eval_params_map,
                            existing_results_files_map=existing_results_files_map,
                            dataset_descriptor=dataset_descriptor,
                            eigen3_svd_service=eigen3_svd_service,
                            model_type='svdpp')


def visualize_hpo_space(rmse_results: pd.DataFrame,
                        fig: FigureBase = None,
                        ax: Axes = None
                        ) -> None:
    """Plots model's RMSE values with different hyperparameters as a colormesh.
    If either fig or ax are not passed, they will be created by this function.

    :param rmse_results: Pandas DataFrame with model's evaluation results
        (in format returned by `__eval_hpo_space` function or its wrappers)
    :param fig: Optional, matplotlib figure to depict the plot.
    :param ax: Optional, matplotlib axes to depict the plot.
    :return: None
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)

    reg_weight_list = rmse_results.index.unique(level='reg_weight')
    n_factors_list = rmse_results.index.unique(level='n_factors')
    y_mesh, x_mesh = np.meshgrid(
        reg_weight_list,
        n_factors_list
    )

    z = rmse_results.to_numpy(dtype=np.float64).reshape(
        len(n_factors_list),
        len(reg_weight_list)
    )
    mesh = ax.pcolormesh(x_mesh, y_mesh, z, shading='auto')
    fig.colorbar(mesh, ax=ax)
