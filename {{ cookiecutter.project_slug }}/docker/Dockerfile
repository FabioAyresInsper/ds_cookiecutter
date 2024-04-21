# Development Stage
#FROM nvidia/cuda:12.0.1-cudnn8-devel-ubuntu22.04
#FROM pytorch/pytorch
FROM python:3.11

ARG USERNAME
ARG PROJECT_NAME={{ cookiecutter.project_slug }}

RUN useradd -m $USERNAME && apt-get update
USER $USERNAME

RUN mkdir /home/$USERNAME/$PROJECT_NAME

WORKDIR /home/$USERNAME/$PROJECT_NAME
