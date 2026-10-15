FROM ros:humble-ros-core-jammy

ENV DEBIAN_FRONTEND=noninteractive

USER root

# install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    git \
    python3-colcon-common-extensions \
    python3-colcon-mixin \
    python3-rosdep \
    python3-vcstool \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y apt-utils curl wget git bash-completion build-essential sudo && rm -rf /var/lib/apt/lists/*

ARG UID=1001
ARG GID=1001

RUN groupadd -o -g ${GID} daniel_docker 2>/dev/null || true && \
    useradd -o -m -u ${UID} -g ${GID} -s /bin/bash daniel_docker 2>/dev/null || true && \
    usermod -aG dialout daniel_docker && \
    mkdir -p /home/daniel_docker/ros2_ws/src && \
    chown -R ${UID}:${GID} /home/daniel_docker

ENV HOME=/home/daniel_docker

RUN mkdir -p /home/daniel_docker/ros2_ws/src && \
    chown -R ${UID}:${GID} /home/daniel_docker/ros2_ws


# bootstrap rosdep
RUN rosdep init && \
  rosdep update --rosdistro $ROS_DISTRO

# setup colcon mixin and metadata
RUN colcon mixin add default \
      https://raw.githubusercontent.com/colcon/colcon-mixin-repository/master/index.yaml && \
    colcon mixin update && \
    colcon metadata add default \
      https://raw.githubusercontent.com/colcon/colcon-metadata-repository/master/index.yaml && \
    colcon metadata update

# install ros2 packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-ros-base=0.10.0-1* \
    && rm -rf /var/lib/apt/lists/*

USER daniel_docker



# sudo docker run --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --rm -it --name docker_geograsp --net host --cpuset-cpus="0-11" -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v /dev:/dev -v $(pwd)/Universal_Robots_ROS2_Driver:/home/docker/ros2_ws/src/Universal_Robots_ROS2_Driver -v $(pwd)/Universal_Robots_ROS2_GZ_Simulation:/home/docker/ros2_ws/src/Universal_Robots_ROS2_GZ_Simulation --user=$(id -u $USER):$(id -g $USER) --pid=host --privileged ros2-docker
