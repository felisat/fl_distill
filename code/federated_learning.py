import os, argparse, json, copy, time
from tqdm import tqdm
import torch, torchvision
import numpy as np
from torch.utils.data import DataLoader

import data, models 
import experiment_manager as xpm
from fl_devices import Client, Server

np.set_printoptions(precision=4, suppress=True)

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)

import gc



parser = argparse.ArgumentParser()
parser.add_argument("--schedule", default="main", type=str)
parser.add_argument("--start", default=0, type=int)
parser.add_argument("--end", default=None, type=int)
parser.add_argument("--reverse_order", default=False, type=bool)
parser.add_argument("--hp", default=None, type=str)

parser.add_argument("--DATA_PATH", default=None, type=str)
parser.add_argument("--RESULTS_PATH", default=None, type=str)
parser.add_argument("--CHECKPOINT_PATH", default=None, type=str)
parser.add_argument("--SHARE_PATH", default=None, type=str)
parser.add_argument("--WORKERS", default=1, type=int)



args = parser.parse_args()



def run_experiment(xp, xp_count, n_experiments, args, seed=0):

  logger.debug(xp)
  hp = xp.hyperparameters
  
  model_fn, optimizer, optimizer_hp = models.get_model(hp["net"])
  if "local_optimizer" in hp:
    optimizer = getattr(torch.optim, hp["local_optimizer"][0])
    optimizer_hp = hp["local_optimizer"][1]

  optimizer_fn = lambda x : optimizer(x, **{k : hp[k] if k in hp else v for k, v in optimizer_hp.items()}) 
  train_data, test_data = data.get_data(hp["dataset"], args.DATA_PATH)
  all_distill_data = data.get_data(hp["distill_dataset"], args.DATA_PATH)

  np.random.seed(seed)
  # What fraction of the unlabeled data should be used for training the anomaly detector
  distill_data = data.IdxSubset(all_distill_data, np.random.permutation(len(all_distill_data))[:hp["n_distill"]]) # data used for distillation
  distill_loader = torch.utils.data.DataLoader(distill_data, batch_size=128, shuffle=True)

  client_data, label_counts = data.get_client_data(train_data, n_clients=hp["n_clients"], 
        classes_per_client=hp["classes_per_client"])

  if hp["aggregation_mode"] in ["FD+S", "FAD+S", "FAD+P+S"]:
    public_data = data.IdxSubset(all_distill_data, np.random.permutation(len(all_distill_data))[hp["n_distill"]:len(all_distill_data)]) # data used to train the outlier detector
    public_loader = torch.utils.data.DataLoader(public_data, batch_size=128, shuffle=True)

    logger.debug("Using {} public data points for distillation and {} public data points for local training.\n".format(len(distill_data), len(public_data)))

    client_loaders = [data.DataMerger({'base': local_data, 'public': public_data}, **hp) for local_data in client_data]

  else:
    client_loaders = [data.DataMerger({'base': local_data}, **hp) for local_data in client_data]

  test_loader = data.DataMerger({'base': data.IdxSubset(test_data, list(range(len(test_data))))}, mixture_coefficients={'base':1}, batch_size=100)
  distill_loader = DataLoader(distill_data, batch_size=128, shuffle=True)

  clients = [Client(model_fn, optimizer_fn, loader, idnum=i, counts=counts, distill_loader=distill_loader) for i, (loader , counts) in enumerate(zip(client_loaders, label_counts))]
  server = Server(model_fn, lambda x : getattr(torch.optim, hp["distill_optimizer"][0])(x, **hp["distill_optimizer"][1]), test_loader, distill_loader)

  # Modes that use pretrained representation 
  if hp["aggregation_mode"] in ["FAD+P", "FAD+P+S"] and os.path.isfile(args.CHECKPOINT_PATH+hp["pretrained"]) :
    for device in clients+[server]:
      device.model.load_state_dict(torch.load(args.CHECKPOINT_PATH+hp["pretrained"], map_location='cpu'), strict=False)
    logger.debug(f"Successfully loaded model from {hp['pretrained']}")


  """
  # Train shallow Outlier detectors
  if hp["aggregation_mode"] in ["FAD+S", "FAD+P+S"]:
    feature_extractor = model_fn().cuda()
    feature_extractor.load_state_dict(torch.load(args.CHECKPOINT_PATH+hp["pretrained"], map_location='cpu'), strict=False)
    feature_extractor.eval()
    for client in clients:
      client.feature_extractor = feature_extractor

    print("Train Outlier Detectors")
    for client in tqdm(clients):
      client.train_outlier_detector(hp["outlier_model"][0], distill_loader, **hp["outlier_model"][1])
  """

  averaging_stats = {"accuracy" : 0.0}
  eval_results = None
  models.print_model(server.model)

  # Start Distributed Training Process
  logger.info("Start Distributed Training..\n")
  t1 = time.time()

  xp.log({"server_val_{}".format(key) : value for key, value in server.evaluate().items()})
  for c_round in range(1, hp["communication_rounds"]+1):

    participating_clients = server.select_clients(clients, hp["participation_rate"])
    xp.log({"participating_clients" : np.array([client.id for client in participating_clients])})

    for client in tqdm(participating_clients):
      client.synchronize_with_server(server, c_round)

      train_stats = client.compute_weight_update(hp["local_epochs"], train_oulier_model=hp["aggregation_mode"] in ["FAD+S", "FAD+P+S"], c_round=c_round,
                max_c_round=hp["communication_rounds"], **hp) 
      logger.debug(train_stats)

    if hp["aggregation_mode"] in ["FA", "FAD", "FAD+P", "FAD+S", "FAD+P+S"]:
      server.aggregate_weight_updates(participating_clients)

      averaging_stats = server.evaluate()
      xp.log({"parameter_averaging_{}".format(key) : value for key, value in averaging_stats.items()})
    
    if hp["aggregation_mode"] in ["FD", "FAD", "FAD+P", "FAD+S", "FAD+P+S"]:

      distill_mode = hp["distill_mode"] if hp["aggregation_mode"] in ["FD+S", "FAD+S", "FAD+P+S"] else "mean_logits"
      distill_stats = server.distill(participating_clients, hp["distill_epochs"], mode=distill_mode, acc0=averaging_stats["accuracy"], fallback=hp["fallback"])
      xp.log({"distill_{}".format(key) : value for key, value in distill_stats.items()})


    # Logging
    if xp.is_log_round(c_round):
      logger.debug("Experiment: {} ({}/{})".format(args.schedule, xp_count+1, n_experiments))   
      
      xp.log({'communication_round' : c_round, 'epochs' : c_round*hp['local_epochs']})
      xp.log({key : clients[0].optimizer.__dict__['param_groups'][0][key] for key in optimizer_hp})
      
      # Evaluate
      eval_results = server.evaluate()
      xp.log({"server_val_{}".format(key) : value for key, value in eval_results.items()})

      # Save results to Disk
      try:
        xp.save_to_disc(path=args.RESULTS_PATH, name=hp['log_path'])
      except:
        logger.error("Saving results Failed!")

      # Timing
      e = int((time.time()-t1)/c_round*(hp['communication_rounds']-c_round))
      logger.debug(f"Remaining Time (approx.): {e // 3600:02d}:{e % 3600 // 60:02d}:{e % 60:02d} [{c_round/hp['communication_rounds']*100:.2f}%]\n")

  # Save model to disk
  server.save_model(path=args.CHECKPOINT_PATH, name=hp["save_model"])
    
  try:
    return 1 - eval_results['accuracy'] if eval_results else 1 - server.evaluate()['accuracy']
  finally:
    # Delete objects to free up GPU memory
    del server; clients.clear()
    del clients
    del xp
    gc.collect()
    #torch.cuda.empty_cache()



def run():

  experiments_raw = json.loads(args.hp)


  hp_dicts = [hp for x in experiments_raw for hp in xpm.get_all_hp_combinations(x)][args.start:args.end]
  if args.reverse_order:
    hp_dicts = hp_dicts[::-1]

  experiments = [xpm.Experiment(hyperparameters=hp) for hp in hp_dicts]

  logger.info("Running {} Experiments..\n".format(len(experiments)))
  for xp_count, experiment in enumerate(experiments):
    run_experiment(experiment, xp_count, len(experiments), args)



if __name__ == "__main__":
  run()
    