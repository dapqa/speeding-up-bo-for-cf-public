import os
from typing import Optional, List

import docker
from docker.client import DockerClient
from docker.models.containers import Container, ExecResult
from docker.types import Mount

from parameters import EnvParameters


class Eigen3SVDService(object):
    """Service class for working with Dockerized native Eigen3 SVD/SVD++ models.
    Requires running Docker, directory with compilable model sources, and
    valid configuration passed via :py:class:`EnvParameters`.
    Native model directory is mounted to Docker root filesystem.
    All the data files must be in the according subdirectory of it.

    This service automatically build needed image, create a container and run it.
    Main methods are :py:meth:`init_container`, :py:meth:`compile_eigen3_svd`, and :py:meth:`run_eigen3_svd`, which are for
    initializing image and container, compiling and running native models respectively.
    """

    def __init__(self,
                 env_params: EnvParameters,
                 do_init_container: bool = True,
                 allow_auto_build_image: bool = False,
                 allow_auto_create_container: bool = False,
                 allow_auto_run_container: bool = False
                 ):
        """Creates an instance of this service and connects to Docker via Docker API.
        If `do_init_container` is True,  :py:meth:`init_container` is called in constructor.

        :param env_params: Environment parameters for the usage of a native model
        :param do_init_container: If True, container initialization is performed in constructor
        :param allow_auto_build_image:  If True, this service will automatically
            create the image if it is not presented
        :param allow_auto_create_container: If True, this service will automatically
            create the container if it is not presented
        :param allow_auto_run_container: If true, this service will automatically
            run the container if it is not running
        """
        self.env_params = env_params
        self.allow_auto_build_image = allow_auto_build_image
        self.allow_auto_create_container = allow_auto_create_container
        self.allow_auto_run_container = allow_auto_run_container

        self.__client: DockerClient = docker.from_env()
        if do_init_container:
            self.init_container()

    def __try_find_container(self) -> Optional[Container]:
        """Tries to find a container with specified in :py:attr:`env_params` name.
        :return: The container or None if it is not found
        """
        container_name = self.env_params.eigen3_svd_container_name

        for existing_container in self.__client.containers.list(all=True):
            if existing_container.name == container_name:
                return existing_container

        return None

    def __image_exists(self) -> bool:
        """Checks if an image with specified in :py:attr:`env_params` tag exists.
        :return: True if the image exists
        """
        for image in self.__client.images.list(all=True):
            if self.env_params.eigen3_svd_image_tag in image.tags:
                return True

        return False

    def __get_container(self,
                        create_if_not_exists: bool = False,
                        build_image_if_not_exists: bool = False,
                        run_if_not_running: bool = True
                        ) -> Container:
        """Returns a running container to interact with.
        If the container or its image does not exists, or the container is not running, this could be
        solved automatically. This behavior is controlled by flags in arguments.

        :param create_if_not_exists: If True, automatic container creation is allowed
        :param build_image_if_not_exists: If True, automatic image build is allowed
        :param run_if_not_running: If True, automatic container run is allowed
        :return: The running container
        """
        container: Container = self.__try_find_container()
        if container is None:
            if create_if_not_exists:
                self.__create_new_container(build_image_if_not_exists=build_image_if_not_exists)
                return self.__get_container(create_if_not_exists=False,
                                            build_image_if_not_exists=False,
                                            run_if_not_running=run_if_not_running)
            else:
                raise RuntimeError(f'Docker container named {self.env_params.eigen3_svd_container_name} not found')

        if container.status != 'running':
            if run_if_not_running:
                container.start()
            else:
                raise RuntimeError('Docker container is not running')

        return container

    def __create_new_container(self, build_image_if_not_exists: bool = False) -> None:
        """Creates new container with specified in :py:attr:`env_params` name from an image
        with specified in :py:attr:`env_params` tag.
        If the image does not exist, it can be built automatically.

        :param build_image_if_not_exists: If true, automatic image build is allowed
        :return: None
        """
        if not self.__image_exists():
            if build_image_if_not_exists:
                self.__build_image()
            else:
                raise RuntimeError(f'Docker image {self.env_params.eigen3_svd_image_tag} does not exist')

        self.__client.containers.create(
            image=self.env_params.eigen3_svd_image_tag,
            name=self.env_params.eigen3_svd_container_name,
            mounts=[
                Mount(
                    target=self.env_params.eigen3_svd_container_dir,
                    source=self.env_params.eigen3_svd_host_dir,
                    type='bind'
                )
            ],
            detach=True,
            tty=True
        )

    def __build_image(self) -> None:
        """Builds an image with specified in :py:attr:`env_params` tag.
        :return: None
        """
        dockerfile_path = os.path.join(self.env_params.eigen3_svd_host_dir, 'docker', 'prod')

        self.__client.images.build(path=dockerfile_path,
                                   tag=self.env_params.eigen3_svd_image_tag)

    def init_container(self) -> None:
        """Performs container initialization according to parameters of this instance.
        :return: None
        """
        self.__get_container(create_if_not_exists=self.allow_auto_create_container,
                             build_image_if_not_exists=self.allow_auto_build_image,
                             run_if_not_running=self.allow_auto_run_container)

    def compile_eigen3_svd(self, n_factors: int, executable_output_name: str) -> None:
        """Compiles a native model from its sources, using the compiler inside the container.

        :param n_factors: Number of factors
        :param executable_output_name: Name of an executable
        :return: None
        """
        container = self.__get_container()

        with open(os.path.join(self.env_params.eigen3_svd_host_dir, 'config.h'), 'w') as config_header:
            config_header.write(f'static const int N_LATENT = {n_factors};\n')

        container.exec_run(
            'make clean',
            workdir=self.env_params.eigen3_svd_container_dir
        )

        container.exec_run(
            'cmake -DCMAKE_BUILD_TYPE=Release',
            workdir=self.env_params.eigen3_svd_container_dir
        )

        make_res = container.exec_run(
            'make',
            workdir=self.env_params.eigen3_svd_container_dir
        )
        if make_res.exit_code != 0:
            raise RuntimeError(make_res.output.decode('utf8'))

        container.exec_run(
            f'mv ./svdistic ./{executable_output_name}',
            workdir=self.env_params.eigen3_svd_container_dir
        )

    def run_eigen3_svd(self, executable_name: str, arguments: List[str]) -> ExecResult:
        """Runs an executable with given arguments inside the Docker container.

        :param executable_name: Name of an executable
        :param arguments: Arguments (e.g. ['a=1', 'b=2'])
        :return: A tuple of exit code and run output
        """
        container = self.__get_container()

        return container.exec_run(
            f'./{executable_name} {" ".join(arguments)}',
            workdir=self.env_params.eigen3_svd_container_dir
        )
