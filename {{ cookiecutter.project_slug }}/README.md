# Autoencoders



## A bit of MLOps

The problems we are having here are:

1. We have (at least) two developers
1. There is a large dataset (not *biiiig data* but around 200GB - it is not a trivial dataset)
1. We have to run the training/test routines in three different machines (not at the same time, of course)
1. We have to keep track of the experiment source code as we progress with tests!

Using our DockerFile, what we can do is the following:

## Setup the development environment

First, make sure you follow the instructions to [install Docker](https://docs.docker.com/desktop/install/linux-install/).

Then, proceed to build the development environment image.

**Warning: this image will contain your SSH PRIVATE KEYS, so DO NOT PUSH IT INTO DOCKERHUB!**

```
docker build . -t autoencoders:dev --build-arg USERNAME=your_username --build-arg SSH_PRV_KEY="$(cat ~/.ssh/id_rsa)"
```

This will setup a development environment with:

* All dependencies installed via `pip`
* All development files (already linked to git) are in: `/home/$USER/dev/autoencoders`

If you want to use GPUs, you will need to follow [NVIDIA's instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) in the *host* (not within the container) so that the container can communicate with GPUs.



## Running the development container

First, we need to start the container as a daemon so it will not close when we exit. The command to start the container will also inform where is the dataset we are using so that it can be mounted on `/mnt/data`.

After that, we need to enter the container in iteractive mode so that we can actually run things:

```
docker run -it -d --gpus all  --name autoencoders_dev --mount type=bind,source=/path/to/your/dataset,target=/mnt/data autoencoders:dev
docker exec -it  autoencoders_dev bash
```

## Running training from outside

To run the training script in the background, so you can close the terminal, use (from host):

```
docker exec -it  autoencoders_dev python code/train.py
```

You will receive messages, but closing the terminal will not shutdown the process.

To stop the process, first find the PID of the training task:

```
docker exec -it autoencoders_dev top
```

Then, kill it with a remote call:

```
docker exec -it autoencoders_dev kill 123
```

## Communicating with the container via SSH

To use SSH tunelling to run code on a remote machine:

```
ssh login@your.ssh.host 'docker exec autoencoders_dev python code/train.py'
```

A long process will keep running even if you close the connection.

In development containers, it could be a good idea to pull the most recent code:

```
ssh login@your.ssh.host 'docker exec autoencoders_dev git pull'
```

## Stop (hard turn off) the container

To stop the container (will kill the process, like shutting down the machine)

```
docker stop autoencoders_dev
```

And, conversely, to start again, use:

```
docker start autoencoders_dev
```


## Install dependencies (deprecated)

```
git clone [this repository]
cd clipaudio
conda env create -f environment.yml
conda activate clipaudio
poetry install
```
