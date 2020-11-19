import logging
import os, argparse, json, copy, time
import warnings
import glob
import hashlib
import logging
import random
logger = logging.getLogger("HPO")
logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings("ignore")


import numpy as np
from ConfigSpace.conditions import InCondition, EqualsCondition
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter

# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from smac.facade.smac_bohb_facade import BOHB4HPO
from smac.initial_design.default_configuration_design import DefaultConfiguration

# Import SMAC-utilities
from smac.scenario.scenario import Scenario

from federated_learning import run_experiment
import experiment_manager as xpm

import torch

# uncool hack since I cannot pass arguments to the experiment wrapper via SMAC
def create_tae_monad(cmdargs):
    def experiment_wrapper(cfg, seed, instance, budget, **kwargs):

        model_name = hashlib.sha1(json.dumps(cfg._values, sort_keys=True).encode()).hexdigest()
        files = glob.glob(f"{args.CHECKPOINT_PATH}/*-{model_name}.pth")
        if files:
            oldbudget = int(files[0].split('/')[-1].split('-')[0])
        else:
            oldbudget = 0

        budget = max(0, int(budget) - oldbudget)
        hp = {
        "dataset" : "mnist", 
        "distill_dataset" : "emnist",
        "net" : "lenet_mnist",

        "n_clients" : 20,
        "classes_per_client" : 0.01,
        "communication_rounds" : budget,
        "participation_rate" : 0.4,
        

        "local_epochs" : cfg["local_epochs"],
        "distill_epochs" : cfg["distill_epochs"],
        "n_distill" : 100000,
        "local_optimizer" : [cfg["local_optimizer"], {"lr" : cfg[f"{cfg['local_optimizer'].lower()}_lr"]}],
        "distill_optimizer" : ["Adam", {"lr" : 0.001}],


        "fallback" : cfg["fallback"],
        "lambda_outlier" : cfg["lambda_outlier"],
        "lambda_fedprox" : cfg["lambda_fedprox"],
        "only_train_final_outlier_layer" : False,
        "warmup_type": "constant",
        "mixture_coefficients" : {"base":cfg["mixture_coefficients_base"], "public":1-cfg["mixture_coefficients_base"]},
        "batch_size" : 256,
        "distill_mode" : "logits_weighted_with_deep_outlier_score",


        "aggregation_mode" : "FAD+P+S",
        

        "pretrained" : f"{oldbudget}-{model_name}.pth",

        "save_model" : f"{budget}-{model_name}.pth",
        "log_frequency" : -100,
        "log_path" : "trash/",
        "job_id" : ['']
        }

        # load already trained model if available

        #hp.update(kwargs)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            experiment = xpm.Experiment(hyperparameters=hp)
            return run_experiment(experiment,0 , 1, cmdargs, seed=seed)
    return experiment_wrapper


def run_hpo(args, tae_runner):
    # create empty config space
    cs = ConfigurationSpace()
    #
    local_epochs = UniformIntegerHyperparameter("local_epochs", 1, 20, default_value=10)
    distill_epochs = UniformIntegerHyperparameter("distill_epochs", 1, 20, default_value=1)

    #
    fallback = CategoricalHyperparameter("fallback", [True, False], default_value=True)
    lambda_outlier = UniformFloatHyperparameter("lambda_outlier", 0.0, 10.0, default_value=1.0)
    lambda_fedprox = UniformFloatHyperparameter("lambda_fedprox", 0.000001, 10.0, default_value=0.01, log=True)

    #
    mixture_coefficients_base = UniformFloatHyperparameter("mixture_coefficients_base", 0.0, 1.0, default_value=0.5)

    #
    local_optimizer = CategoricalHyperparameter("local_optimizer", ["Adam", "SGD"], default_value="Adam")
    adam_lr = UniformFloatHyperparameter("adam_lr", 0.00001, 1.0, default_value=0.001, log=True)
    sgd_lr = UniformFloatHyperparameter("sgd_lr", 0.00001, 1.0, default_value=0.1, log=True)

    cs.add_hyperparameters([local_epochs, distill_epochs, fallback, lambda_outlier, lambda_fedprox, mixture_coefficients_base, local_optimizer, adam_lr, sgd_lr])


    use_adam_lr = EqualsCondition(child=adam_lr, parent=local_optimizer, value='Adam')
    use_sgd_lr = EqualsCondition(child=sgd_lr, parent=local_optimizer, value='SGD')
    cs.add_conditions([use_adam_lr, use_sgd_lr])

    #


    # Scenario object
    scenario = Scenario({
        "run_obj": "quality",  # we optimize quality (alternatively runtime)
        "runcount-limit": 100,  # max. number of function evaluations; for this example set to a low number
        "cs": cs,  # configuration space
        "deterministic": True,
        "shared_model": True,
        "input_psmac_dirs": args.SHARE_PATH + '*',
        "output_dir": args.SHARE_PATH,
        "limit_resources": False
    })

    max_iters = 50
    # intensifier parameters
    intensifier_kwargs = {'initial_budget': 5, 'max_budget': max_iters, 'eta': 3}

    # Optimize, using a SMAC-object
    print("Optimizing!")
    smac = BOHB4HPO(
        scenario=scenario, 
        tae_runner=tae_runner,
        intensifier_kwargs=intensifier_kwargs,
        n_jobs=args.WORKERS,
    )

    # Start optimization
    try:
        incumbent = smac.optimize()
    finally:
        incumbent = smac.solver.incumbent

    inc_value = smac.get_tae_runner().run(config=incumbent, instance='2',
                                          budget=1, seed=0)[1]
    print("Optimized Value: %.4f" % inc_value)

if __name__ == '__main__':
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
    torch.multiprocessing.set_start_method('forkserver')

    run_hpo(args, create_tae_monad(args))
