import os, argparse, json, copy, time
from tqdm import tqdm
import torch, torchvision
import numpy as np

import data, models
import experiment_manager as xpm
from fl_devices import Client, Server

#import torch.multiprocessing
#torch.multiprocessing.set_sharing_strategy('file_system')

#torch.autograd.set_detect_anomaly(True)
#torch.set_num_threads(1)


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

    # print hyperparameters
    print(xp)
    hp = xp.hyperparameters

    # create model and optimizer
    model_fn, optimizer, optimizer_hp = models.get_model(hp["net"])
    optimizer_fn = lambda x: optimizer(
        x, **{k: hp[k] if k in hp else v for k, v in optimizer_hp.items()}
    )

    # create data
    train_data, test_data = data.get_data(hp["dataset"], args.DATA_PATH)
    distill_data = data.get_data(hp["distill_dataset"], args.DATA_PATH)
    distill_data = torch.utils.data.Subset(
        distill_data, np.random.permutation(len(distill_data))[: hp["n_distill"]]
    )
    distill_data_raw = torch.stack([x for x, y in distill_data], dim=0)

    # get data loaders
    client_loaders, test_loader, client_data, test_data = data.get_loaders(
        train_data,
        test_data,
        n_clients=hp["n_clients"],
        classes_per_client=hp["classes_per_client"],
        batch_size=int(hp["batch_size"] * hp["local_data_percentage"])
        if hp["distill_phase"] == "clients"
        else hp["batch_size"],
        n_data=None,
    )
    distill_loader = torch.utils.data.DataLoader(
        distill_data, batch_size=128, shuffle=False
    )

    # generate clients and server
    clients = [
        Client(
            model_fn,
            optimizer_fn,
            loader,
            client_dataset=(
                client_data[i] if hp["distill_phase"] == "clients" else None
            ),
        )
        for i, loader in enumerate(client_loaders)
    ]
    server = Server(
        model_fn,
        lambda x: torch.optim.Adam(x, lr=0.001, weight_decay=5e-4),
        test_loader,
        distill_loader,
        clients=clients,
        aux_data=distill_data_raw,
        **hp,
    )
    # server.load_model(path=args.CHECKPOINT_PATH, name=hp["pretrained"])

    if hp["pretrained"]:
        for device in clients + [server]:
            device.model.load_state_dict(
                torch.load(
                    args.CHECKPOINT_PATH + hp["pretrained"][hp["distill_dataset"]],
                    map_location="cpu",
                ),
                strict=False,
            )
        print("Successfully loader model from", hp["pretrained"][hp["distill_dataset"]])

    if hp["only_linear"]:
        for device in [server] + clients:
            for param in device.model.f.parameters():
                param.requires_grad = False

    # print model
    models.print_model(server.model)

    # Start Distributed Training Process
    print("Start Distributed Training..\n")
    t1 = time.time()

    xp.log({f"server_val_{key}": value for key, value in server.evaluate().items()})
    for c_round in range(1, hp["communication_rounds"] + 1):

        participating_clients = server.select_clients(clients, hp["participation_rate"])

        train_stats_clients = []
        for client in tqdm(participating_clients):
            client.synchronize_with_server(server)
            # client.generate_feature_bank()

            train_stats = client.compute_weight_update(
                hp["local_epochs"],
                c_round=c_round,
                max_c_round=hp["communication_rounds"],
                warmup_type=hp["warmup_type"],
                distill_weight=hp["distill_weight"],
            )
            #train_stats_clients.append(train_stats)

        # log client epoch and minibatch loss
        #xp.log(
        #    {
        #        f"train_local_loss": torch.stack(
        #            [stat["detailed_loss"] for stat in train_stats_clients]
        #        )
        #    },
        #    printout=False,
        #)

        # use FA or if normal client distillation aggregate on server
        if hp["aggregation_mode"] in ["FA", "FAD"] or hp["distill_phase"] == 'clients':
            server.aggregate_weight_updates(participating_clients, distill_phase=hp["distill_phase"], aggregation_mode=hp["aggregation_mode"])

        if hp["aggregation_mode"] in ["FD", "FAD", "FknnD"]:

            distill_stats = server.distill(
                participating_clients,
                hp["distill_epochs"],
                mode=hp["distill_mode"],
                distill_phase=hp["distill_phase"],
            )
            xp.log(
                {
                    f"distill_{key}": value
                    for key, value in distill_stats.items()
                    if key != "softlabels"
                }
            )

        # Logging
        if xp.is_log_round(c_round):
            print(f"Experiment: {args.schedule} ({xp_count+1}/{n_experiments})")

            xp.log(
                {"communication_round": c_round, "epochs": c_round * hp["local_epochs"]}
            )
            xp.log(
                {
                    key: clients[0].optimizer.__dict__["param_groups"][0][key]
                    for key in optimizer_hp
                }
            )

            # Evaluate
            xp.log(
                {f"server_val_{key}": value for key, value in server.evaluate().items()}
            )

            # Save results to Disk
            try:
                xp.save_to_disc(path=args.RESULTS_PATH, name=hp["log_path"])
            except:
                print("Saving results Failed!")

            # Timing
            e = int(
                (time.time() - t1) / c_round * (hp["communication_rounds"] - c_round)
            )
            print(
                "Remaining Time (approx.):",
                f"{(e // 3600):02d}:{(e % 3600 // 60):02d}:{(e % 60):02d}",
                f"[{(c_round/hp['communication_rounds']*100):.2f}%]\n",
            )

    # Save model to disk
    server.save_model(path=args.CHECKPOINT_PATH, name=hp["save_model"])

    # Delete objects to free up GPU memory
    del server
    clients.clear()
    torch.cuda.empty_cache()


def run():

    experiments_raw = json.loads(args.hp)

    hp_dicts = [hp for x in experiments_raw for hp in xpm.get_all_hp_combinations(x)][
        args.start : args.end
    ]
    if args.reverse_order:
        hp_dicts = hp_dicts[::-1]
    experiments = [xpm.Experiment(hyperparameters=hp) for hp in hp_dicts]

    print("Running {} Experiments..\n".format(len(experiments)))
    for xp_count, experiment in enumerate(experiments):
        run_experiment(experiment, xp_count, len(experiments))


if __name__ == "__main__":
    run()
