# Speeding up Bayesian Optimization of Matrix Factorization Recommender Models Hyperparameters

Implementation of a Bayesan Opimization method that uses Gaussian Process with 
modified kernel for speeding up hyperparameter optimization process for 
Matrix Factorization models.

This repository contains both kernel's and experiments' source code from the corresponding paper.

## Usage

Here are step-by-step instruction for reproducing the presented results.  
Tools that must be installed before are:
- Git
- Docker
- Anaconda. [Miniconda](https://docs.conda.io/en/latest/miniconda.html) is a quick way to get started.

### 1. Download the project

This project contains submodules, which are essential for running native SVD/SVD++ models.  
If you have Git 2.13+:

```shell
git clone --recurse-submodules https://github.com/dapqa/speeding-up-bo-for-cf-public.git
```

If you have an older version of Git:

```shell
git clone https://github.com/dapqa/speeding-up-bo-for-cf-public.git
cd speeding-up-bo-for-cf-public
git submodule update --init --recursive
```

### 2. Create an environment

Run the following command to create a conda environment: 

```shell
conda env create -n speeding-up-bo-for-cf-env -f environment.yaml
```

To activate it:

```shell
conda env activate speeding-up-bo-for-cf-env
```

### 3. Download datasets

The experiments are performed on [MovieLens](https://grouplens.org/datasets/movielens/) datasets, 
which are licensed to [GroupLens](https://grouplens.org/).

To automatically download these datasets from the original distribution and preprocess them, run:

```shell
python -m util.datasets
```

### 4. Run experiments

All the experiments are in Jupyter Notebooks in the `notebooks` directory.
To start a Jupyter server, run:

```shell
jupyter notebook
```

There will be a link to open in a browser.
Open it, navigate to the `notebooks` directory, open and run needed notebooks.

All the results are saved in the `results` directory, so the evaluation
is not performed and results are just loaded from CSV files.
If you want to re-run an experiment, just delete or rename corresponding result files.

### Docker usage note

Native SVD/SVD++ models which are used in experiments runs in a Docker container.
An image can be automatically built from the `eigen3_svd/docker/prod/Dockerfile`, 
and a container can also be automatically created and run.

Mounts, names, and tags configuration for docker is in `parameters.py` file and 
also can be modified via command-line parameters if needed.

If it is not ok that container manipulation is automated, this behavior can be
modified directly in experiments' notebooks. If automatic creation or running is turned off,
it must be performed manually.