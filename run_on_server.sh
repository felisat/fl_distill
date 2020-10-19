#!/bin/bash

max=$1
step=$2

git add *
git commit -m "run experiment on cluster"
git push

ssh -o "StrictHostKeyChecking no" fsattler@vca-gpu-211-01 << EOF
	cd fl_distill
	git pull
	head -n 50 exec.sh
	echo "Running $max experiments..."

	for (( c=0; c<$max; c+=$step ))
	do
	   echo sbatch exec.sh "--start $c --end $(($c+$step))"
	   #sbatch exec.sh "--start $c --end $(($c+$step))"
	done
EOF