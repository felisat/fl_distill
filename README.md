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
      
      
 ## Hyperparameters
 
 ### Task
- `"dataset"` : Choose from `["mnist", "cifar10"]`
- `"net"` : Choose from `["mobilenetv2", "lenet_mnist", "lenet_cifar", "vgg11", "vgg11s"]`

### Federated Learning Environment

- `"n_clients"` : Number of Clients
- `"classes_per_client"` : Number of different Classes every Client holds in it's local data, 0 returns an iid split
- `"participation_rate"` : Fraction of Clients which participate in every Communication Round
- `"batch_size"` : Batch-size used by the Clients
- `"balancedness"` : Default 1.0, if <1.0 data will be more concentrated on some clients
- `"communication_rounds"` : Total number of communication rounds

### Logging 
- `"log_frequency"` : Number of communication rounds after which results are logged and saved to disk
- `"log_path"` : e.g. "results/experiment1/"

Run multiple experiments by listing different configurations.


