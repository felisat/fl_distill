import torch, torchvision
import numpy as np

def get_mnist(path):
  transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                                torchvision.transforms.ToTensor(),    
                                                torchvision.transforms.Normalize((0.1307,), (0.3081,))])
  train_data = torchvision.datasets.MNIST(root=path+"MNIST", train=True, download=True, transform=transforms)
  test_data = torchvision.datasets.MNIST(root=path+"MNIST", train=False, download=True, transform=transforms)

  return train_data, test_data

def get_cifar10(path):
  transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                          torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                               (0.2023, 0.1994, 0.2010))])
  train_data = torchvision.datasets.CIFAR10(root=path+"CIFAR", train=True, download=True, transform=transforms)
  test_data = torchvision.datasets.CIFAR10(root=path+"CIFAR", train=False, download=True, transform=transforms)

  return train_data, test_data


def get_data(dataset, path):
  return {"cifar10" : get_cifar10, "mnist" : get_mnist}[dataset](path)

def get_loaders(train_data, test_data, n_clients=10, classes_per_client=0, batch_size=128, n_data=None):

  subset_idcs = split_image_data(train_data.targets, n_clients, classes_per_client, n_data)
  client_data = [torch.utils.data.Subset(train_data, subset_idcs[i]) for i in range(n_clients)]
 
  client_loaders = [torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True) for subset in client_data]
  test_loader = torch.utils.data.DataLoader(test_data, batch_size=100)

  return client_loaders, test_loader



def split_image_data(labels, n_clients, classes_per_client, n_data):

  if isinstance(labels, torch.Tensor):
    labels = labels.numpy()
  if not n_data: 
    n_data = len(labels)
  n_labels = np.max(labels) + 1
  label_idcs = {l : np.argwhere(np.array(labels[:n_data])==l).flatten().tolist() for l in range(n_labels)}

  if classes_per_client == 0:
    idcs = np.random.permutation(n_data//n_clients*n_clients).reshape(n_clients, -1)
  else:
    data_per_client = n_data // n_clients
    data_per_client_per_class = data_per_client // classes_per_client

    idcs = []
    for i in range(n_clients):
      client_idcs = []
      budget = data_per_client
      c = np.random.randint(n_labels)
      while budget > 0:
        take = min(data_per_client_per_class, len(label_idcs[c]), budget)
        
        client_idcs += label_idcs[c][:take]
        label_idcs[c] = label_idcs[c][take:]
        
        budget -= take
        c = (c + 1) % n_labels

      idcs += [client_idcs]


  print_split(idcs, labels)

  return idcs


def print_split(idcs, labels):
  n_labels = np.max(labels) + 1 
  print("Data split:")
  for i, idccs in enumerate(idcs):
    split = np.sum(np.array(labels)[idccs].reshape(1,-1)==np.arange(n_labels).reshape(-1,1), axis=1)
    print(" - Client {}: {}".format(i,split), flush=True)
  print()