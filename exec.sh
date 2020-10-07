#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tim.korjakow@hhi.fraunhofer.de
#SBATCH --output=out/%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1


cmdargs=$1

hyperparameters=' [{
	"dataset" : ["cifar10"], 
	"distill_dataset" : ["stl10"],
	"net" : ["vgg11s"],
	

	"n_clients" : [2],
	"classes_per_client" : [0.1],
	"balancedness" : [1.0],


	"communication_rounds" : [2],
	"participation_rate" : [1],
	"local_epochs" : [2],
	"distill_epochs" : [10],
	"n_distill" : [1000],
	"warmup_type": ["constant"],

	
	"batch_size" : [128],
	"local_data_percentage" : [0.05],
	"distill_weight": [1],
	"aggregation_mode" : ["FD"],
	"distill_mode" : ["regular"],
	"distill_phase" : ["server","clients"],
	"only_linear" : [false],
	

	"pretrained" : [null],

	"save_model" : [null],
	"log_frequency" : [-100],
	"log_path" : ["noniid/"],
	"job_id" : [['$SLURM_JOB_ID']]}]'



if [[ "$HOSTNAME" == *"vca"* ]]; then # Cluster

	RESULTS_PATH="/opt/small_files/"
	DATA_PATH="/opt/in_ram_data/"
	CHECKPOINT_PATH="/opt/checkpoints/"

	echo $hyperparameters
	source "/etc/slurm/local_job_dir.sh"

	export SINGULARITY_BINDPATH="$LOCAL_DATA:/data,$LOCAL_JOB_DIR:/mnt/output,./code:/opt/code,./checkpoints:/opt/checkpoints,./results:/opt/small_files,$HOME/in_ram_data:/opt/in_ram_data"
	singularity exec --nv $HOME/base_images/pytorch15.sif python -u /opt/code/federated_learning.py --hp="$hyperparameters" --RESULTS_PATH="$RESULTS_PATH" --DATA_PATH="$DATA_PATH" --CHECKPOINT_PATH="$CHECKPOINT_PATH" $cmdargs

	mkdir -p results
	cp -r ${LOCAL_JOB_DIR}/. ${SLURM_SUBMIT_DIR}/results	


else # Local

	RESULTS_PATH="results/"
	DATA_PATH="data/"
	CHECKPOINT_PATH="checkpoints/"
	echo "$hyperparameters"

	python -u code/federated_learning.py --hp="$hyperparameters" --RESULTS_PATH="$RESULTS_PATH" --DATA_PATH="$DATA_PATH" --CHECKPOINT_PATH="$CHECKPOINT_PATH" $cmdargs




fi






