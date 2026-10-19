FROM ros:humble

# ============================================================
# Base configuration
# ============================================================

USER root

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

# ============================================================
# Install build tools, Python and ROS 2 dependencies
# ============================================================

RUN apt-get update && apt-get install --no-install-recommends -y \
    apt-utils \
    build-essential \
    nano \
    cmake \
    git \
    curl \
    wget \
    ca-certificates \
    bash-completion \
    sudo \
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-colcon-common-extensions \
    python3-colcon-mixin \
    python3-colcon-ros \
    python3-rosdep \
    python3-vcstool \
    ros-humble-ur \
    ros-humble-ros-base \
    ros-humble-ament-cmake \
    ros-humble-ros2cli \
    ros-humble-rosidl-default-generators \
    ros-humble-angles \
    ros-humble-controller-interface \
    ros-humble-controller-manager \
    ros-humble-hardware-interface \
    ros-humble-joint-state-broadcaster \
    ros-humble-joint-trajectory-controller \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \ 
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install ros-humble-ur-msgs


# RUN sed -i '/find_package(PythonInterp 3.6 REQUIRED)/i \
# if(POLICY CMP0148)\n\
#   cmake_policy(SET CMP0148 OLD)\n\
# endif()\n' \
#     /opt/ros/humble/share/rosidl_generator_py/cmake/rosidl_generator_py_generate_interfaces.cmake


# Install CMake 3.28.3
RUN wget -q \
    https://github.com/Kitware/CMake/releases/download/v3.28.3/cmake-3.28.3-linux-x86_64.tar.gz \
    -O /tmp/cmake.tar.gz \
    && tar -xzf /tmp/cmake.tar.gz -C /opt \
    && mv /opt/cmake-3.28.3-linux-x86_64 /opt/cmake-3.28.3 \
    && rm /tmp/cmake.tar.gz

ENV PATH="/opt/cmake-3.28.3/bin:${PATH}"

# ============================================================
# Create non-root user
# ============================================================

ARG UID=1000
ARG GID=1000

RUN groupadd -o -g ${GID} daniel_docker 2>/dev/null || true && \
    useradd -o -m -u ${UID} -g ${GID} -s /bin/bash daniel_docker 2>/dev/null || true && \
    usermod -aG dialout daniel_docker && \
    mkdir -p /home/daniel_docker/ros2_ws/src && \
    chown -R ${UID}:${GID} /home/daniel_docker

ENV HOME=/home/daniel_docker

# ============================================================
# Initialize rosdep
# ============================================================

RUN rosdep init 2>/dev/null || true && \
    rosdep update --rosdistro humble

# ============================================================
# Configure colcon mixins and metadata
# ============================================================

RUN colcon mixin add default \
        https://raw.githubusercontent.com/colcon/colcon-mixin-repository/master/index.yaml && \
    colcon mixin update && \
    colcon metadata add default \
        https://raw.githubusercontent.com/colcon/colcon-metadata-repository/master/index.yaml && \
    colcon metadata update

# ============================================================
# Workspace configuration
# ============================================================

WORKDIR /home/daniel_docker/ros2_ws

# Instalar UR Gazebo
RUN cd src/ && git clone -b humble https://github.com/UniversalRobots/Universal_Robots_ROS2_Gazebo_Simulation.git && \
    rosdep update && rosdep install --ignore-src --from-paths . -y

# Instalar UR Driver
# RUN cd src && 
# Source ROS 2 Humble and create the MPC package
RUN chown -R ${UID}:${GID} /home/daniel_docker/ros2_ws


# ============================================================
# Environment configuration
# ============================================================

USER daniel_docker

RUN echo "source /opt/ros/humble/setup.bash" >> /home/daniel_docker/.bashrc && \
echo "if [ -f /home/daniel_docker/ros2_ws/install/setup.bash ]; then source /home/daniel_docker/ros2_ws/install/setup.bash; fi" >> /home/daniel_docker/.bashrc

CMD ["/bin/bash"]

RUN . /opt/ros/humble/setup.sh && \
    cd /home/daniel_docker/ros2_ws/src && \
    ros2 pkg create --build-type ament_python \
        mpc

COPY ./config/mpc/package.xml /home/daniel_docker/ros2_ws/src/mpc/ 
COPY ./config/mpc/setup.py /home/daniel_docker/ros2_ws/src/mpc/ 



# Set workspace ownership after creating the package
# BUILD: sudo docker build     --no-cache     --build-arg UID="$(id -u)"     --build-arg GID="$(id -g)"     -t ros2-docker .
# LAUNCH: sudo docker run --gpus all     --shm-size=1g     --ulimit memlock=-1     --ulimit stack=67108864     --rm -it     --name ros2_docker     --net host     --cpuset-cpus="0-11"     -e DISPLAY="$DISPLAY"     -v /tmp/.X11-unix:/tmp/.X11-unix:rw     -v /dev:/dev   -v ./mpc:/home/daniel_docker/ros2_ws/src/mpc/mpc       --user="$(id -u):$(id -g)"     --pid=host     --privileged     ros2-docker
# FIX PERMISSIONS: sudo docker exec -u root ros2_docker     chown -R daniel_docker:daniel_docker /home/daniel_docker/.ros
