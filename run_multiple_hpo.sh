#!/bin/bash

max=$1

head -n 50 exec.sh
echo "Running $max experiments..."

for (( c=0; c<$max; c+=1 ))
do
   sbatch poetry run bash exec.sh
done
