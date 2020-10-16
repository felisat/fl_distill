#!/bin/bash

git add *
git commit -m "run experiment on cluster"
git push

ssh -o "StrictHostKeyChecking no" fsattler@vca-gpu-211-01 << EOF
	cd fl_distill
	git pull
	tail -n 100 exec.sh
EOF