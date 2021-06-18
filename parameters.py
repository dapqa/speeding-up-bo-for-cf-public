from argparse import ArgumentParser
from dataclasses import dataclass
import os


@dataclass
class EnvParameters:
    eigen3_svd_host_dir: str
    eigen3_svd_container_dir: str
    eigen3_svd_image_tag: str
    eigen3_svd_container_name: str


def __create_argparser() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument(
        '--eigen3_svd_host_dir', type=str, default='./eigen3_svd',
        help='Path to Eigen3 SVD dir at host machine'
    )
    parser.add_argument(
        '--eigen3_svd_container_dir', type=str, default='/eigen3_svd',
        help='Path to mounted Eigen3 SVD dir inside Docker container'
    )
    parser.add_argument(
        '--eigen3_svd_image_tag', type=str, default='eigen3-svd/prod',
        help='Tag of Eigen3 SVD Docker image'
    )
    parser.add_argument(
        '--eigen3_svd_container_name', type=str, default='eigen3-svd-prod',
        help='Name of Eigen3 SVD Docker container'
    )

    return parser


def __validate_and_fix_env_parameters(env_params: EnvParameters) -> None:
    if not os.path.exists(env_params.eigen3_svd_host_dir):
        raise ValueError(f'Path {env_params.eigen3_svd_host_dir} does not exist')

    env_params.eigen3_svd_host_dir = os.path.abspath(env_params.eigen3_svd_host_dir)


def get_env_parameters() -> EnvParameters:
    parser = __create_argparser()

    params_namespace, _ = parser.parse_known_args()
    params = EnvParameters(**vars(params_namespace))

    __validate_and_fix_env_parameters(params)
    return params


if __name__ == '__main__':
    print(get_env_parameters())
