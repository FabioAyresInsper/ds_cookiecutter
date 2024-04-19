# Development Stage
FROM nvidia/cuda:12.0.1-cudnn8-devel-ubuntu22.04
FROM pytorch/pytorch

ARG USERNAME
ARG SSH_PRV_KEY

ARG GIT_REPO=git@github.com:tiagoft/autoencoders.git
ARG PROJECT_NAME=autoencoders

COPY requirements.txt .

# Configure and install everything
RUN useradd -m $USERNAME \
    && apt-get update \
    && apt-get install -y wget git openssh-server ca-certificates libsndfile-dev graphviz \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && pip install -r requirements.txt

USER $USERNAME

RUN mkdir /home/$USERNAME/.ssh \
    && chmod 0700 /home/$USERNAME/.ssh \
    && echo "$SSH_PRV_KEY" > /home/$USERNAME/.ssh/id_rsa \
    && chmod 600 /home/$USERNAME/.ssh/id_rsa \
    && eval `ssh-agent -s` \
    && ssh-keyscan github.com >> /home/$USERNAME/.ssh/known_hosts \
    && ssh-add /home/$USERNAME/.ssh/id_rsa \
    && mkdir /home/$USERNAME/dev \
    && chown -hR $USERNAME /home/$USERNAME \
    && cd /home/$USERNAME/dev \
    && git clone $GIT_REPO

WORKDIR /home/$USERNAME/dev/$PROJECT_NAME
