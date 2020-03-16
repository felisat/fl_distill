import os, argparse, json, copy, time
from tqdm import tqdm
import torch, torchvision
import numpy as np

import data, models 
import experiment_manager as xpm
from fl_devices import Client, Server

if "vca" in os.popen('hostname').read().rstrip(): # Runs on Cluster
  CODE_PATH = "/opt/code/"
  CHECKPOINT_PATH = "/opt/checkpoints/"
  RESULTS_PATH = "/opt/small_files/"
  DATA_PATH = "/opt/in_ram_data/"
else:
  CODE_PATH = ""
  RESULTS_PATH = "/home/sattler/Workspace/PyTorch/Remote/fl_base/results/"
  DATA_PATH = "/home/sattler/Data/PyTorch/"
  CHECKPOINT_PATH = "/home/sattler/Workspace/PyTorch/Remote/fl_base/checkpoints/"


np.set_printoptions(precision=4, suppress=True)

parser = argparse.ArgumentParser()
parser.add_argument("--schedule", default="main", type=str)
parser.add_argument("--start", default=0, type=int)
parser.add_argument("--end", default=None, type=int)
parser.add_argument("--reverse_order", default=False, type=bool)
parser.add_argument("--hp", default=None, type=str)
args = parser.parse_args()



def run_experiments(experiments):
  print("Running {} Experiments..\n".format(len(experiments)))
  for xp_count, xp in enumerate(experiments):
    print(xp)
    hp = xp.hyperparameters
    
    model_fn, optimizer, optimizer_hp = models.get_model(hp["net"])
    optimizer_fn = lambda x : optimizer(x, **{k : hp[k] if k in hp else v for k, v in optimizer_hp.items()}) 
    train_data, test_data = data.get_data(hp["dataset"], DATA_PATH)

    client_loaders, test_loader = data.get_loaders(train_data, test_data, n_clients=hp["n_clients"], 
          classes_per_client=hp["classes_per_client"], batch_size=hp["batch_size"], n_data=hp["n_data"])

    clients = [Client(model_fn, optimizer_fn, loader) for loader in client_loaders]
    server = Server(model_fn, test_loader)
    server.load_model(path=CHECKPOINT_PATH, name=hp["pretrained"])

    # print model
    models.print_model(server.model)

    # Start Distributed Training Process
    print("Start Distributed Training..\n")
    t1 = time.time()
    for c_round in range(1, hp["communication_rounds"]+1):

      participating_clients = server.select_clients(clients, hp["participation_rate"])
      
      for client in (participating_clients):
        client.synchronize_with_server(server)
        train_stats = client.compute_weight_update(hp["local_epochs"])  
        client.reset()
        
      server.aggregate_weight_updates(clients)


      # Logging
      if xp.is_log_round(c_round):
        print("Experiment: {} ({}/{})".format(args.schedule, xp_count+1, len(experiments)))   
        
        xp.log({'communication_round' : c_round, 'lr' : clients[0].optimizer.__dict__['param_groups'][0]['lr'],
          'epochs' : c_round*hp['local_epochs']})
        
        # Evaluate  
        xp.log({"client_train_{}".format(key) : value for key, value in train_stats.items()})
        xp.log({"server_val_{}".format(key) : value for key, value in server.evaluate().items()})

        # Save results to Disk
        try:
          xp.save_to_disc(path=RESULTS_PATH, name=hp['log_path'])
        except:
          print("Saving results Failed!")

        # Timing
        e = int((time.time()-t1)/c_round*(hp['communication_rounds']-c_round))
        print("Remaining Time (approx.):", '{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60), 
                  "[{:.2f}%]\n".format(c_round/hp['communication_rounds']*100))

    # Save model to disk
    server.save_model(path=CHECKPOINT_PATH, name=hp["save_model"])
      
    # Delete objects to free up GPU memory
    del server; clients.clear()
    torch.cuda.empty_cache()


def run():

  if args.hp:
    experiments_raw = json.loads(args.hp)
  else:
    with open(CODE_PATH+'federated_learning.json') as data_file:    
      experiments_raw = json.load(data_file)[args.schedule]


  hp_dicts = [hp for x in experiments_raw for hp in xpm.get_all_hp_combinations(x)][args.start:args.end]
  if args.reverse_order:
    hp_dicts = hp_dicts[::-1]
  experiments = [xpm.Experiment(hyperparameters=hp) for hp in hp_dicts]

  run_experiments(experiments)

if __name__ == "__main__":
  # Load the Hyperparameters of all Experiments to be performed and set up the Experiments
  run()
    