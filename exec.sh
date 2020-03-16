#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=felix.sattler@hhi.fraunhofer.de
#SBATCH --output=out/%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1


hyperparameters=' [{
	"dataset" : ["cifar10"], 
	"net" : ["lenet_cifar"],
	
	"n_clients" : [1],
	"n_data" : [1000],

	"communication_rounds" : [100],
	"local_epochs" : [1],

	"participation_rate" : [1.0],
	
	"classes_per_client" : [0],
	"batch_size" : [128],
	"lr" : [0.01],
	"balancedness" : [1.0],

	"swipe" : [{"lr" : ["e10", "float", 0.1, 0.001], "batch_size" : ["e2", "int", 8, 512], "weight_decay" : ["e10", "float", 1e-6, 1e-2]}],

	"pretrained" : [null],
	"save_model" : [null],
	"log_frequency" : [-100],
	"log_path" : ["test_hyperparameter2/"],
	"job_id" : [['$SLURM_JOB_ID']]}]'



if [[ "$HOSTNAME" == *"vca"* ]]; then # Cluster
	echo $hyperparameters
	source "/etc/slurm/local_job_dir.sh"

	export SINGULARITY_BINDPATH="$LOCAL_DATA:/data,$LOCAL_JOB_DIR:/mnt/output,./code:/opt/code,./results:/opt/small_files,$HOME/in_ram_data:/opt/in_ram_data"
	singularity exec --nv $HOME/base_images/full.sif python -u /opt/code/federated_learning.py --hp="$hyperparameters"

	mkdir -p results
	cp -r ${LOCAL_JOB_DIR}/. ${SLURM_SUBMIT_DIR}/results	


else # Local
	source activate dcfl
	python -u code/federated_learning.py --hp="$hyperparameters"




fi






