#!/bin/bash

cmdargs=$1

head -n 50 exec.sh
echo "Running $cmdargs experiments..."

for (( c=0; c<$cmdargs; c++ ))
do
   sbatch exec.sh "--start $c --end $(($c+1))"
done