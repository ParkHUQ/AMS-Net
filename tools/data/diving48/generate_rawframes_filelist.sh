#!/usr/bin/env bash

cd ../../../
PYTHONPATH=. python tools/data/build_file_list.py diving48 /home/dataset/video/RGB/Diving48/Diving48_frames/ --num-split 1 --level 1 --subset train --format rawframes --rgb-prefix frames
PYTHONPATH=. python tools/data/build_file_list.py diving48 /home/dataset/video/RGB/Diving48/Diving48_frames/ --num-split 1 --level 1 --subset val --format rawframes   --rgb-prefix frames
echo "Filelist for rawframes generated."

cd tools/data/diving48/
