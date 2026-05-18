FROM ubuntu:jammy-20260509

ENV DEBIAN_FRONTEND=noninteractive

ARG UID=1000
ARG GID=1000

RUN apt-get update && apt-get install -y \
    apt-utils \
    curl \
    wget \
    git \
    bash-completion \
    build-essential \
    sudo \
 && rm -rf /var/lib/apt/lists/*

# Python
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
 && rm -rf /var/lib/apt/lists/*

# Make python -> python3
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install PyTorch (CPU version)
RUN pip install torch torchvision



# Create user
RUN addgroup --gid ${GID} lfd && \
    adduser --gecos "LFD User" \
    --disabled-password \
    --uid ${UID} \
    --gid ${GID} lfd

RUN usermod -a -G dialout lfd

# Passwordless sudo
RUN echo "lfd ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/99_lfd && \
    chmod 0440 /etc/sudoers.d/99_lfd

# Install pip bootstrap
RUN curl https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py

# NVIDIA support
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

# User home
ENV HOME=/home/lfd

RUN mkdir -p ${HOME} && \
    chown -R lfd:lfd ${HOME}

WORKDIR ${HOME}

USER lfd

CMD ["/bin/bash"]

# sudo docker build -t docker_lfd .
# sudo docker run --gpus all -it   --env DISPLAY=$DISPLAY   --volume /tmp/.X11-unix:/tmp/.X11-unix   -v .:/home/lfd docker_lfd
