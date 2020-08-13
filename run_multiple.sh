#!/bin/bash

cmdargs=$1

for (( c=0; c<$cmdargs; c++ ))
do
   sbatch exec.sh "--start $c --end $c+1"
done