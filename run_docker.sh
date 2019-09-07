#!/usr/bin/env bash

DATA_ROOT=${GLOW_DATA_ROOT:-/media/yuthon/Data/}
RESULT_DIR=${GLOW_RESULT_DIR:-/media/yuthon/Data/result}
DOCKER_IMAGE=corenel/pytorch:cu80-pytorch0.2.0

docker run --rm -it --init \
  --runtime=nvidia \
  --ipc=host \
  --network=host \
  --volume=$PWD:/app/code \
  --volume=${DATA_ROOT}:/app/data \
  --volume=${RESULT_DIR}:/app/result \
  --volume=$HOME/.torch:/home/user/.cache/torch \
  --volume=$PWD/.ssh:/home/user/.ssh \
  -e LC_ALL=C.UTF-8 \
  -e LANG=C.UTF-8 \
  -e LOCAL_UID=$(id -u ${USER}) \
  -e LOCAL_GID=$(id -g ${USER}) \
  -e CUDA_VISIBLE_DEVICES=${1} \
  ${DOCKER_IMAGE} \
  "${@:2}"
