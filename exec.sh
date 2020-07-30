#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=felix.sattler@hhi.fraunhofer.de
#SBATCH --output=out/%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1


hyperparameters=' [{
	"dataset" : ["cifar10"], 
	"distill_dataset" : ["stl10"],
	"net" : ["vgg11s"],
	
	"n_clients" : [20],

	"communication_rounds" : [60],
	"local_epochs" : [20],
	"distill_epochs" : [10],
	"n_distill" : [100, 1000, 10000, 100000], 

	"participation_rate" : [0.4],
	
	"classes_per_client" : [0],
	"batch_size" : [128],

	"use_distillation" : [true],
	"aggregate" : [false],
	"compress" : [true],


	"balancedness" : [1.0],

	"pretrained" : [null],
	"save_model" : [null],
	"log_frequency" : [-100],
	"log_path" : ["n_distill/"],
	"job_id" : [['$SLURM_JOB_ID']]}]'



if [[ "$HOSTNAME" == *"vca"* ]]; then # Cluster
	echo $hyperparameters
	source "/etc/slurm/local_job_dir.sh"

	export SINGULARITY_BINDPATH="$LOCAL_DATA:/data,$LOCAL_JOB_DIR:/mnt/output,./code:/opt/code,./checkpoints:/opt/checkpoints,./results:/opt/small_files,$HOME/in_ram_data:/opt/in_ram_data"
	singularity exec --nv $HOME/base_images/pytorch15.sif python -u /opt/code/federated_learning.py --hp="$hyperparameters"

	mkdir -p results
	cp -r ${LOCAL_JOB_DIR}/. ${SLURM_SUBMIT_DIR}/results	


else # Local
	python -u code/federated_learning.py --hp="$hyperparameters"




fi






