# Federated Learning with Model Distillation on the vca cluster


## Usage

### Run on local machine

1.) In `exec.sh` define paths

	RESULTS_PATH="results/"
	DATA_PATH="/path/to/where/you/store/your/datasets"
	CHECKPOINT_PATH="checkpoints/"
  
2.) and set the hyperparameters
  
    hyperparameters="[{...}]"

3.) Run via

    bash exec.sh

### Run on vca cluster


1.) In your home directory on the server, create a folder `base_images/`, containing the file `pytorch15.def`:

	<<<<<<<<<<<< pytorch15.def >>>>>>>>>>>
	Bootstrap: docker
	From: pytorch/pytorch:latest

	%post
	export "PATH=/opt/conda/bin:$PATH"

	conda install matplotlib
	conda install numpy
	conda install scipy
	conda install tqdm
	<<<<<<<<<<<< pytorch15.def >>>>>>>>>>>

2.) Run

    singularity build --force --fakeroot pytorch15.sif pytorch15.def
    
    
3.) Create a folder `in_ram_data/` where you save your data sets (CIFAR, MNIST, ..)

4.) Change email address:

    #SBATCH --mail-user=your.mail@hhi.fraunhofer.de

4.) Run via

      sbatch exec.sh
     
5.) Check if everything is working    

      watch tail -n 100 out/<SLURM_JOB_ID>.out 
