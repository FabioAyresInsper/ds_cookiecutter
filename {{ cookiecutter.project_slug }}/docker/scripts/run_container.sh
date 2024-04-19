#!/bin/bash
docker run \
    -it \
    --mount type=bind,source=/path/to/your/dataset,target=/mnt/data \
    --name $1 \
    $2
    
