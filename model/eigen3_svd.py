"""Classes for using dockerized native SVD/SVD++ models.
Their interface is similar to common Python model classes for the easier usage.
"""

import re
import os
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import List

from parameters import EnvParameters, get_env_parameters
from util.docker.eigen3_svd import Eigen3SVDService
from util.splits import DatasetSplitFile


@dataclass
class PredictionFile:
    """Contains info about output predictions file in the native models directory"""
    source: DatasetSplitFile
    output_file_path: str


class AbstractEigen3SVDModel(metaclass=ABCMeta):
    """Base class for dockerized native SVD/SVD++ models.
    Implements the whole set of needed operations, and its children differs only in commands used for calling a model.
    """

    def __init__(self,
                 env_params: EnvParameters,
                 eigen3_svd_service: Eigen3SVDService,
                 model_id: str,
                 n_factors: int = 100,
                 lr: float = 0.05,
                 n_epochs: int = 20,
                 reg_bias: float = 0.02,
                 reg_weight: float = 0.02,
                 lr_decay: float = 1.0
                 ):
        """Constructor.

        :param env_params: Environment parameters for the usage of a native model
        :param eigen3_svd_service: Docker service for a model
        :param model_id: Unique ID for saving model dump
        :param n_factors: Number of factors
        :param lr: Learning rate
        :param n_epochs: Number of training epochs
        :param reg_bias: Regularization constant for biases
        :param reg_weight: Regularization constant for weights
        :param lr_decay: Learning rate decay (per epoch)
        """
        self.env_params = env_params
        self.eigen3_svd_service = eigen3_svd_service
        self.model_id = model_id
        self.n_factors = n_factors
        self.lr = lr
        self.n_epochs = n_epochs
        self.reg_bias = reg_bias
        self.reg_weight = reg_weight
        self.lr_decay = lr_decay

    @staticmethod
    def __get_executable_name(n_factors: int):
        """Returns name of a specific executable file for given number of factors.

        :param n_factors: Number of factors
        :return: Executable file name
        """
        return f'svdistic{n_factors}'

    @staticmethod
    def compile(factor_counts_list: List[int],
                eigen3_svd_service: Eigen3SVDService,
                recompile_if_exists: bool = False,
                verbose: bool = True
                ) -> List[str]:
        """Compiles multiple executables, one per given number of factors.

        :param factor_counts_list: List with desired numbers of factors
        :param eigen3_svd_service: Docker service
        :param recompile_if_exists: If true and an executable already exists, it will be recompiled
        :param verbose: If true, method execution process will be printed
        :return: List of output executable names (ordered as in `factor_counts_list` parameter)
        """
        executable_names = []

        for n_factors in factor_counts_list:
            executable_name = AbstractEigen3SVDModel.__get_executable_name(n_factors)
            executable_path = os.path.join(
                eigen3_svd_service.env_params.eigen3_svd_host_dir,
                executable_name
            )

            if os.path.exists(executable_path):
                if not recompile_if_exists:
                    if verbose:
                        print(f'Executable for factor count: {n_factors} already exists, skipped')

                    executable_names.append(executable_name)
                    continue
                else:
                    if verbose:
                        print(f'Executable for factor count: {n_factors} already exists, but will be recompiled')

            eigen3_svd_service.compile_eigen3_svd(n_factors, executable_name)
            if verbose:
                print(f'Executable for factor count: {n_factors} has been complied')

            executable_names.append(executable_name)

        return executable_names

    @abstractmethod
    def _get_alg_name(self) -> str:
        """Returns command line argument for calling a specific matrix factorization algorithm
            (svd for SVD, svdpp for SVD++)
        """
        pass

    def __create_or_find_executable(self) -> str:
        """Tries to find an executable for number of factors that has been set for this instance.
        If the executable does not exist, compiles it.

        :return: Name of the executable to use
        """
        executable_names = AbstractEigen3SVDModel.compile([self.n_factors], self.eigen3_svd_service,
                                                          recompile_if_exists=False, verbose=False)
        return executable_names[0]

    def __create_container_data_file_path(self, host_path: str) -> str:
        """Creates relative path to a data file inside the native models directory in a Docker container
            from the same host path.
        Since the native models directory is mounted from host filesystem, the same host path always exists.

        :param host_path: Path to a data file on a host machine
        :return: Relative path to this data file for a model
        """
        relative_path = os.path.relpath(host_path, os.path.join(self.env_params.eigen3_svd_host_dir, 'data', 'corpus'))
        linux_relative_path = relative_path.replace(os.sep, '/')
        return linux_relative_path

    def fit(self, data: DatasetSplitFile) -> None:
        """Fits a model using given train file.

        :param data: Train file reference
        :return: None
        """
        executable_name = self.__create_or_find_executable()
        data_file_name = self.__create_container_data_file_path(data.path)

        result = self.eigen3_svd_service.run_eigen3_svd(
            executable_name,
            [
                self._get_alg_name(),
                'train',
                f'-model_id {self.model_id}',
                f'-n_epochs {self.n_epochs}',
                '-report_freq 1',
                f'-fname {data_file_name}',
                f'-n_user {data.descriptor.max_user_id + 1}',
                f'-n_product {data.descriptor.max_item_id + 1}',
                f'-n_example {data.n_rows}',
                f'-lr {self.lr}',
                f'-reg_bias {self.reg_bias}',
                f'-reg_weight {self.reg_weight}',
                f'-lr_decay {self.lr_decay}'
            ])

        if result.exit_code != 0:
            raise RuntimeError(result.output)

    def rmse(self, data: DatasetSplitFile) -> float:
        """Calculates RMSE of model predictions on given test file.

        :param data: Test file reference
        :return: RMSE
        """
        executable_name = self.__create_or_find_executable()
        data_file_name = self.__create_container_data_file_path(data.path)

        result = self.eigen3_svd_service.run_eigen3_svd(
            executable_name,
            [
                self._get_alg_name(),
                'score',
                f'-model_id {self.model_id}',
                f'-fname {data_file_name}',
                f'-n_user {data.descriptor.max_user_id + 1}',
                f'-n_product {data.descriptor.max_item_id + 1}',
                f'-n_example {data.n_rows}'
            ]
        )

        if result.exit_code != 0:
            raise RuntimeError(result.output)

        rmse_row = next(s for s in result.output.decode('utf8').split('\n') if s.startswith('RMSE'))
        parsed_rmse = re.findall("\\d+\\.\\d+", rmse_row)
        if len(parsed_rmse) > 0:
            return float(parsed_rmse[0])
        else:
            return float('nan')

    def predict(self, data: DatasetSplitFile) -> PredictionFile:
        """Predicts ratings for given input file.
        The result is saved into a separate file.

        :param data: Input file reference
        :return: Reference to a file that contains predicted ratings
        """
        executable_name = self.__create_or_find_executable()
        data_file_name = self.__create_container_data_file_path(data.path)

        result = self.eigen3_svd_service.run_eigen3_svd(
            executable_name,
            [
                self._get_alg_name(),
                'infer',
                f'-model_id {self.model_id}',
                f'-fname {data_file_name}',
                f'-n_user {data.descriptor.max_user_id + 1}',
                f'-n_product {data.descriptor.max_item_id + 1}',
                f'-n_example {data.n_rows}'
            ]
        )

        if result.exit_code != 0:
            raise RuntimeError(result.output)

        return PredictionFile(
            data, os.path.join(
                self.env_params.eigen3_svd_host_dir,
                'data', 'saves', f'{self.model_id}-inferred.txt'
            )
        )


class Eigen3SVD(AbstractEigen3SVDModel):
    """Class for using a dockerized native SVD model."""

    def __init__(self,
                 env_params: EnvParameters,
                 eigen3_svd_service: Eigen3SVDService,
                 model_id: str,
                 n_factors: int = 100,
                 lr: float = 0.05,
                 n_epochs: int = 20,
                 reg_bias: float = 0.02,
                 reg_weight: float = 0.02,
                 lr_decay: float = 1.0
                 ):
        """Constructor. See :py:class:`AbstractEigen3SVDModel` for parameters documentation."""
        super().__init__(
            env_params=env_params,
            eigen3_svd_service=eigen3_svd_service,
            model_id=model_id,
            n_factors=n_factors,
            lr=lr,
            n_epochs=n_epochs,
            reg_bias=reg_bias,
            reg_weight=reg_weight,
            lr_decay=lr_decay
        )

    def _get_alg_name(self) -> str:
        return 'svd'


class Eigen3SVDpp(AbstractEigen3SVDModel):
    """Class for using a dockerized native SVD model."""

    def __init__(self,
                 env_params: EnvParameters,
                 eigen3_svd_service: Eigen3SVDService,
                 model_id: str,
                 n_factors: int = 100,
                 lr: float = 0.05,
                 n_epochs: int = 20,
                 reg_bias: float = 0.02,
                 reg_weight: float = 0.02,
                 lr_decay: float = 1.0
                 ):
        """Constructor. See :py:class:`AbstractEigen3SVDModel` for parameters documentation."""
        super().__init__(
            env_params=env_params,
            eigen3_svd_service=eigen3_svd_service,
            model_id=model_id,
            n_factors=n_factors,
            lr=lr,
            n_epochs=n_epochs,
            reg_bias=reg_bias,
            reg_weight=reg_weight,
            lr_decay=lr_decay
        )

    def _get_alg_name(self) -> str:
        return 'svdpp'
