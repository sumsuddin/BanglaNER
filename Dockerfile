FROM continuumio/miniconda3

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y \
    git \
    fish \
    wget \
    make

WORKDIR /tmp

ADD Makefile Makefile
ADD environment.yml environment.yml
ADD requirements.txt requirements.txt

RUN make create_environment

ARG USERNAME=dockeruser
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && usermod -aG video dockeruser

USER $USERNAME

