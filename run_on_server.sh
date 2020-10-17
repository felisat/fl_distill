#!/bin/bash

git add *
git commit -m "run experiment on cluster"
git push

ssh -o "StrictHostKeyChecking no" fsattler@vca-gpu-211-01 << EOF
	cd fl_distill
	git pull
	head -n 50 exec.sh
	sbatch exec.sh
EOF