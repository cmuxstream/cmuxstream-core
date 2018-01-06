#!/usr/bin/env bash

source env/bin/activate

dataset=$1
filenames=`ls /home/localdirs/stufs1/emanzoor/dev/xstream/data/LowDim/$dataset*`
for entry in $filenames
do
  python baseline_dataset_report.py $entry
done
