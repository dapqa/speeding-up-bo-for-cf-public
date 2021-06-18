import os
from dataclasses import dataclass
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd

from parameters import EnvParameters

# The directory for datasets
DATASETS_DIR = 'data'

# Defines a name of a file that contains dataset ratings as CSV with columns
# user_id, item_id, rating [, ... possible others]
RATINGS_FILE_NAME = 'prepared-ratings.csv'


# Dataset descriptors

@dataclass
class DatasetDescriptor:
    """Descriptor with basic info about a dataset (but not data itself)"""
    id: str
    name: str
    url: str
    dir: str
    n_users: int
    n_items: int
    n_rows: int
    max_user_id: int
    max_item_id: int


# Descriptor of a dataset (or its part) saved on disk
@dataclass
class DatasetFile:
    path: str
    descriptor: DatasetDescriptor


MOVIELENS_100K = DatasetDescriptor(
    id='ml100k', name='Movielens 100k',
    url='http://files.grouplens.org/datasets/movielens/ml-100k.zip',
    dir=os.path.join(DATASETS_DIR, 'movielens-100k'),
    n_users=943, n_items=1682, n_rows=100000, max_user_id=943, max_item_id=1682)

MOVIELENS_1M = DatasetDescriptor(
    id='ml1m', name='Movielens 1M',
    url='https://files.grouplens.org/datasets/movielens/ml-1m.zip',
    dir=os.path.join(DATASETS_DIR, 'movielens-1m'),
    n_users=6040, n_items=3706, n_rows=1000209, max_user_id=6040, max_item_id=3952)

MOVIELENS_10M = DatasetDescriptor(
    id='ml10m', name='Movielens 10M',
    url='https://files.grouplens.org/datasets/movielens/ml-10m.zip',
    dir=os.path.join(DATASETS_DIR, 'movielens-10m'),
    n_users=69878, n_items=10677, n_rows=10000054, max_user_id=71567, max_item_id=65133)


# Helper private functions

def _download_and_extract_zip(url, output_dir_name, verbose=True):
    """Downloads a zip file from given url and extracts it into given output directory.
    If the output dir already exists, does nothing.
    """
    if os.path.exists(output_dir_name):
        if verbose:
            print(f'{output_dir_name} already exists, skipped')
        return

    with urlopen(url) as zip_response:
        with ZipFile(BytesIO(zip_response.read())) as zip_file:
            zip_file.extractall(output_dir_name)

    if verbose:
        print(f'Zip file from {url} has been extracted to {output_dir_name}')


def _transform_movielens_100k():
    raw_ratings = pd.read_csv(os.path.join(MOVIELENS_100K.dir, 'ml-100k', 'u.data'),
                              sep='\t',
                              header=None,
                              names=['user_id', 'item_id', 'rating', 'timestamp'])

    raw_ratings.to_csv(os.path.join(MOVIELENS_100K.dir, RATINGS_FILE_NAME),
                       sep=',',
                       index=None)


def _transform_movielens_1m():
    raw_ratings = pd.read_csv(os.path.join(MOVIELENS_1M.dir, 'ml-1m', 'ratings.dat'),
                              sep='::',
                              header=None,
                              names=['user_id', 'item_id', 'rating', 'timestamp'])

    raw_ratings.to_csv(os.path.join(MOVIELENS_1M.dir, RATINGS_FILE_NAME),
                       sep=',',
                       index=None)


def _transform_movielens_10m():
    raw_ratings = pd.read_csv(os.path.join(MOVIELENS_10M.dir, 'ml-10M100K', 'ratings.dat'),
                              sep='::',
                              header=None,
                              names=['user_id', 'item_id', 'rating', 'timestamp'])

    raw_ratings.to_csv(os.path.join(MOVIELENS_10M.dir, RATINGS_FILE_NAME),
                       sep=',',
                       index=None)


# Public functions

def download_and_transform_dataset(dataset_descriptor: DatasetDescriptor, verbose=True):
    """Downloads a dataset by given descriptor.
    Next, transforms it into a CSV file that always contains columns 'user_id', 'item_id', 'rating'. Some additional columns may also
    be presented.
    This file can be found in os.path.join(dataset_descriptor.dir, RATINGS_FILE_NAME)
    """
    _download_and_extract_zip(dataset_descriptor.url, dataset_descriptor.dir, verbose=verbose)

    if os.path.exists(os.path.join(dataset_descriptor.dir, RATINGS_FILE_NAME)):
        print(f'Dataset is already tranformed, skipped')
    else:
        if dataset_descriptor.name == MOVIELENS_100K.name:
            _transform_movielens_100k()
        elif dataset_descriptor.name == MOVIELENS_1M.name:
            _transform_movielens_1m()
        elif dataset_descriptor.name == MOVIELENS_10M.name:
            _transform_movielens_10m()

    if verbose:
        print(f'Dataset "{dataset_descriptor.name}" is ready for use')


# Common dataset readers

def as_pandas(dataset_descriptor: DatasetDescriptor, only_ratings=True) -> pd.DataFrame:
    full_df = pd.read_csv(os.path.join(dataset_descriptor.dir, RATINGS_FILE_NAME), sep=',')
    if only_ratings:
        return full_df[['user_id', 'item_id', 'rating']]
    else:
        return full_df


def as_numpy(dataset_descriptor: DatasetDescriptor, only_ratings=True) -> np.ndarray:
    df = as_pandas(dataset_descriptor, only_ratings)
    return df.to_numpy()


# Runnable part

if __name__ == '__main__':
    download_and_transform_dataset(MOVIELENS_100K)
    download_and_transform_dataset(MOVIELENS_1M)
    download_and_transform_dataset(MOVIELENS_10M)
