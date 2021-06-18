from dataclasses import dataclass
from math import ceil
from typing import Tuple

from sklearn.model_selection import train_test_split

from parameters import EnvParameters
from util.datasets import *


@dataclass
class DatasetSplitFile:
    """Reference to a file containing dataset part"""
    path: str
    descriptor: DatasetDescriptor
    n_rows: int


def split_for_eigen3_svd(dataset_descriptor: DatasetDescriptor,
                         env_params: EnvParameters,
                         train_file_name: str,
                         test_file_name: str,
                         test_size: float,
                         overwrite_if_exists: bool = False
                         ) -> Tuple[DatasetSplitFile, DatasetSplitFile]:
    """Splits given dataset to train and test parts for Eigen 3 SVD/SVD++ native model.
    Saves them on disc in the data directory of native models.
    Data are sorted by `user_id` column.

    :param dataset_descriptor: Descriptor of a dataset
    :param env_params: Environment parameters for the usage of a native model
    :param train_file_name: Output name for a train dataset file
    :param test_file_name: Output name for a test dataset file
    :param test_size: Float value from 0 to 1 defining the proportion of a dataset to include into a test set
    :param overwrite_if_exists: If True and both files are already exist, nothing will be done
    :return: Train and test file references
    """
    output_train_file = os.path.join(env_params.eigen3_svd_host_dir, 'data', 'corpus', train_file_name)
    output_test_file = os.path.join(env_params.eigen3_svd_host_dir, 'data', 'corpus', test_file_name)

    if os.path.exists(output_train_file) and os.path.exists(output_test_file) and not overwrite_if_exists:
        print(f'{output_train_file} and {output_test_file} already exists')
    else:
        df = as_pandas(dataset_descriptor)
        train_df, test_df = train_test_split(df, test_size=test_size)

        train_df = train_df.sort_values(by=['user_id'])
        test_df = test_df.sort_values(by=['user_id'])

        train_df.to_csv(output_train_file, header=None, index=None)
        test_df.to_csv(output_test_file, header=None, index=None)

    return DatasetSplitFile(output_train_file, dataset_descriptor,
                            dataset_descriptor.n_rows - ceil(dataset_descriptor.n_rows * test_size)), \
           DatasetSplitFile(output_test_file, dataset_descriptor,
                            ceil(dataset_descriptor.n_rows * test_size))
