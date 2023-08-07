#!/usr/bin/env bash

CONFIG=$1
#CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# Arguments starting from the forth one are captured by ${@:4}
python -u -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG --launcher pytorch ${@:4}

#echo $CONFIG
#echo $CHECKPOINT
#$CHECKPOINT 