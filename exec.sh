#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=felix.sattler@hhi.fraunhofer.de
#SBATCH --output=out/%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1


hyperparameters=' [{
	"dataset" : ["cifar10"], 
	"net" : ["mobilenetv2s", "mobilenetv2xs", "mobilenetv2", "mobilenetv2_gn"],
	
	"n_clients" : [1, 2, 10],
	"n_data" : [null],

	"communication_rounds" : [100],
	"local_epochs" : [1],

	"participation_rate" : [1.0],
	
	"classes_per_client" : [0],
	"batch_size" : [128],


	"balancedness" : [1.0],

	"swipe" : [{}],

	"pretrained" : [null],
	"save_model" : [null],
	"log_frequency" : [-100],
	"log_path" : ["mobilenets/"],
	"job_id" : [['$SLURM_JOB_ID']]}]'



if [[ "$HOSTNAME" == *"vca"* ]]; then # Cluster
	echo $hyperparameters
	source "/etc/slurm/local_job_dir.sh"

	export SINGULARITY_BINDPATH="$LOCAL_DATA:/data,$LOCAL_JOB_DIR:/mnt/output,./code:/opt/code,./checkpoints:/opt/checkpoints,./results:/opt/small_files,$HOME/in_ram_data:/opt/in_ram_data"
	singularity exec --nv $HOME/base_images/full.sif python -u /opt/code/federated_learning.py --hp="$hyperparameters"

	mkdir -p results
	cp -r ${LOCAL_JOB_DIR}/. ${SLURM_SUBMIT_DIR}/results	


else # Local
	source activate dcfl
	python -u code/federated_learning.py --hp="$hyperparameters"




fi






