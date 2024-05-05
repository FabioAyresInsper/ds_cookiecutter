#% if cookiecutter.base_dev_environment == 'pytorch'

FROM nvidia/cuda:12.0.1-cudnn8-devel-ubuntu22.04
FROM pytorch/pytorch

#% else

FROM python:3.11

#% endif

ARG USERNAME=user
ARG PROJECT_NAME={{ cookiecutter.project_slug }}

RUN useradd -m $USERNAME \
    && apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y coreutils \
    && apt-get install -y sudo \
    && usermod -aG sudo user \
    && echo "$USERNAME ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

USER $USERNAME

RUN mkdir /home/$USERNAME/$PROJECT_NAME
RUN mkdir -p /home/$USERNAME/.local/bin

WORKDIR /home/$USERNAME/$PROJECT_NAME
