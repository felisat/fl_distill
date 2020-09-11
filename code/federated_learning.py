import os, argparse, json, copy, time
from tqdm import tqdm
import torch, torchvision
import numpy as np

import data, models 
import experiment_manager as xpm
from fl_devices import Client, Server


np.set_printoptions(precision=4, suppress=True)

parser = argparse.ArgumentParser()
parser.add_argument("--schedule", default="main", type=str)
parser.add_argument("--start", default=0, type=int)
parser.add_argument("--end", default=None, type=int)
parser.add_argument("--reverse_order", default=False, type=bool)
parser.add_argument("--hp", default=None, type=str)

parser.add_argument("--DATA_PATH", default=None, type=str)
parser.add_argument("--RESULTS_PATH", default=None, type=str)
parser.add_argument("--CHECKPOINT_PATH", default=None, type=str)

args = parser.parse_args()



def run_experiment(xp, xp_count, n_experiments):

  print(xp)
  hp = xp.hyperparameters
  
  model_fn, optimizer, optimizer_hp = models.get_model(hp["net"])
  optimizer_fn = lambda x : optimizer(x, **{k : hp[k] if k in hp else v for k, v in optimizer_hp.items()}) 
  train_data, test_data = data.get_data(hp["dataset"], args.DATA_PATH)
  distill_data = data.get_data(hp["distill_dataset"], args.DATA_PATH)
  distill_data = torch.utils.data.Subset(distill_data, np.random.permutation(len(distill_data))[:hp["n_distill"]])

  client_loaders, test_loader = data.get_loaders(train_data, test_data, n_clients=hp["n_clients"], 
        classes_per_client=hp["classes_per_client"], batch_size=hp["batch_size"], n_data=None)
  distill_loader = torch.utils.data.DataLoader(distill_data, batch_size=128, shuffle=False)

  clients = [Client(model_fn, optimizer_fn, loader) for loader in client_loaders]
  server = Server(model_fn, lambda x : torch.optim.Adam(x, lr=0.001, weight_decay=5e-4), test_loader, distill_loader)
  #server.load_model(path=args.CHECKPOINT_PATH, name=hp["pretrained"])

  if hp["pretrained"]:
    for device in clients+[server]:
      device.model.load_state_dict(torch.load(args.CHECKPOINT_PATH+hp["pretrained"][hp["distill_dataset"]], map_location='cpu'), strict=False)
    print("Successfully loader model from", hp["pretrained"][hp["distill_dataset"]])

  if hp["only_linear"]:
    for device in [server]+clients:
      for param in device.model.f.parameters():
        param.requires_grad = False

  # print model
  models.print_model(server.model)

  # Start Distributed Training Process
  print("Start Distributed Training..\n")
  t1 = time.time()

  xp.log({"server_val_{}".format(key) : value for key, value in server.evaluate().items()})
  for c_round in range(1, hp["communication_rounds"]+1):

    participating_clients = server.select_clients(clients, hp["participation_rate"])

    for client in tqdm(participating_clients):
      client.synchronize_with_server(server)
      #client.generate_feature_bank() 

      train_stats = client.compute_weight_update(hp["local_epochs"]) 

    if hp["aggregation_mode"] in ["FA", "FAD"]:
      server.aggregate_weight_updates(participating_clients)
    
    if hp["aggregation_mode"] in ["FD", "FAD", "FknnD"]:

      #xp.log({"predictions" : clients[0].compute_prediction_matrix(distill_loader)})

      #hist = server.compute_prediction_histogram(participating_clients)
      #print(hist)
      #exit()

      distill_stats = server.distill(participating_clients, hp["distill_epochs"], mode=hp["distill_mode"])
      xp.log({"distill_{}".format(key) : value for key, value in distill_stats.items()})


    # Logging
    if xp.is_log_round(c_round):
      print("Experiment: {} ({}/{})".format(args.schedule, xp_count+1, n_experiments))   
      
      xp.log({'communication_round' : c_round, 'epochs' : c_round*hp['local_epochs']})
      xp.log({key : clients[0].optimizer.__dict__['param_groups'][0][key] for key in optimizer_hp})
      
      # Evaluate  
      #xp.log({"client_train_{}".format(key) : value for key, value in train_stats.items()})
      #xp.log({"client_val_{}".format(key) : value for key, value in client.evaluate(server.loader).items()})
      xp.log({"server_val_{}".format(key) : value for key, value in server.evaluate().items()})

      # Save results to Disk
      try:
        xp.save_to_disc(path=args.RESULTS_PATH, name=hp['log_path'])
      except:
        print("Saving results Failed!")

      # Timing
      e = int((time.time()-t1)/c_round*(hp['communication_rounds']-c_round))
      print("Remaining Time (approx.):", '{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60), 
                "[{:.2f}%]\n".format(c_round/hp['communication_rounds']*100))

  # Save model to disk
  server.save_model(path=args.CHECKPOINT_PATH, name=hp["save_model"])
    
  # Delete objects to free up GPU memory
  del server; clients.clear()
  torch.cuda.empty_cache()


def run():


  experiments_raw = json.loads(args.hp)


  hp_dicts = [hp for x in experiments_raw for hp in xpm.get_all_hp_combinations(x)][args.start:args.end]
  if args.reverse_order:
    hp_dicts = hp_dicts[::-1]
  experiments = [xpm.Experiment(hyperparameters=hp) for hp in hp_dicts]

  print("Running {} Experiments..\n".format(len(experiments)))
  for xp_count, experiment in enumerate(experiments):
    run_experiment(experiment, xp_count, len(experiments))


if __name__ == "__main__":
  run()
    