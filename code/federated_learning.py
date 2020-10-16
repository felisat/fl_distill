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
  distill_data = data.IdxSubset(distill_data, np.random.permutation(len(distill_data))[:hp["n_distill"]])

  client_loaders, test_loader, label_counts = data.get_loaders(train_data, test_data, n_clients=hp["n_clients"], 
        classes_per_client=hp["classes_per_client"], batch_size=hp["batch_size"], n_data=None)
  distill_loader = torch.utils.data.DataLoader(distill_data, batch_size=128, shuffle=False)

  clients = [Client(model_fn, optimizer_fn, loader, idnum=i) for i, loader in enumerate(client_loaders)]
  server = Server(model_fn, lambda x : torch.optim.SGD(x, lr=0.01, momentum=0.9), test_loader, distill_loader)
  #server.load_model(path=args.CHECKPOINT_PATH, name=hp["pretrained"])
  #server.model.load_state_dict(torch.load("/home/sattler/Workspace/PyTorch/fl_distill/checkpoints/simclr_net_bn_stl10_80epochs.pth", map_location='cpu'), strict=False)

  for client, counts in zip(clients, label_counts):
    client.label_counts = counts

  if hp["pretrained"]:
    for device in clients+[server]:
      device.model.load_state_dict(torch.load(args.CHECKPOINT_PATH+hp["pretrained"], map_location='cpu'), strict=False)
    print("Successfully loader model from", hp["pretrained"])

  if hp["only_linear"]:
    for device in [server]+clients:
      for param in device.model.f.parameters():
        param.requires_grad = False

  if "n_adversaries" in hp:
    for client in clients:
      if client.id<hp["n_adversaries"]:
        client.is_adversary = True
      else:
        client.is_adversary = False



  if hp["distill_mode"] == "outlier_score":
    #feature_extractor = models.lenet_large().cuda()
    #feature_extractor.load_state_dict(torch.load(args.CHECKPOINT_PATH+"simclr_lenet_stl10_10epochs.pth", map_location='cpu'), strict=False)
    #feature_extractor.eval()
    #for client in clients:
    #  client.feature_extractor = feature_extractor

    print("Train Outlier Detectors")
    for client in tqdm(clients):
      client.train_outlier_detector(hp["outlier_model"][0], distill_loader, **hp["outlier_model"][1])

  # print model
  models.print_model(server.model)

  # Start Distributed Training Process
  print("Start Distributed Training..\n")
  t1 = time.time()

  xp.log({"server_val_{}".format(key) : value for key, value in server.evaluate().items()})
  for c_round in range(1, hp["communication_rounds"]+1):

    participating_clients = server.select_clients(clients, hp["participation_rate"])
    xp.log({"participating_clients" : np.array([c.id for c in participating_clients])})

    for client in tqdm(participating_clients):
      client.synchronize_with_server(server, c_round)
      #client.generate_feature_bank() 

      train_stats = client.compute_weight_update(hp["local_epochs"]) 

    if hp["aggregation_mode"] in ["FA", "FAD"]:
      server.aggregate_weight_updates(participating_clients)
    
    if hp["aggregation_mode"] in ["FD", "FAD", "FknnD"]:

      #xp.log({"predictions" : np.stack([client.compute_prediction_matrix(distill_loader) for client in participating_clients])})

      #hist = server.compute_prediction_histogram(participating_clients)
      #print(hist)
      #exit()

      distill_stats = server.distill(clients, hp["distill_epochs"], mode=hp["distill_mode"])
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
    