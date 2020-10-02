import torch, torchvision
import numpy as np
from itertools import cycle

def get_mnist(path):

  class AddChannels(object):
    def __init__(self, n_channels=3):
      self.n_channels = n_channels
    def __call__(self, x):
      return torch.cat([x]*self.n_channels, dim=0)


  transforms = torchvision.transforms.Compose([ torchvision.transforms.Resize((32,32)),
                                                torchvision.transforms.ToTensor(),    
                                                AddChannels()
                                                #torchvision.transforms.Normalize((0.1307,), (0.3081,))])
                                                ])
  train_data = torchvision.datasets.MNIST(root=path+"MNIST", train=True, download=True, transform=transforms)
  test_data = torchvision.datasets.MNIST(root=path+"MNIST", train=False, download=True, transform=transforms)

  return train_data, test_data

def get_cifar10(path):
  transforms = torchvision.transforms.Compose([
                                          #torchvision.transforms.RandomCrop(32, padding=4),
                                          #torchvision.transforms.RandomHorizontalFlip(),
                                          torchvision.transforms.ToTensor(),
                                          torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                               (0.2023, 0.1994, 0.2010))
                                          ])
  train_data = torchvision.datasets.CIFAR10(root=path+"CIFAR", train=True, download=True, transform=transforms)
  test_data = torchvision.datasets.CIFAR10(root=path+"CIFAR", train=False, download=True, transform=transforms)

  return train_data, test_data


def get_stl10(path):
  transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                               torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                               (0.2023, 0.1994, 0.2010))
                                               ])

  data = torchvision.datasets.STL10(root=path+"STL10", split='unlabeled', folds=None, 
                             transform=transforms,
                                    download=True)
  return data

def get_svhn(path):
  transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                               torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                               (0.2023, 0.1994, 0.2010))
                                               ])

  data = torchvision.datasets.SVHN(root=path+"SVHN", split='train', transform=transforms,
                                    download=True)
  return data


def get_data(dataset, path):
  return {"cifar10" : get_cifar10, "mnist" : get_mnist, "stl10" : get_stl10, "svhn" : get_svhn}[dataset](path)

def get_loaders(train_data, test_data, n_clients=10, classes_per_client=0, batch_size=128, n_data=None):

  # one-hot encode labels
  # TODO: all labels seem to start with 1 upwards ?
  one_hot_matrix = torch.eye(torch.max(torch.tensor(train_data.targets))+1)
  train_data.targets = [one_hot_matrix[label] for label in train_data.targets]
  test_data.targets = [one_hot_matrix[label] for label in test_data.targets]

  subset_idcs = split_dirichlet(train_data.targets, n_clients, n_data, classes_per_client)
  client_data = [torch.utils.data.Subset(train_data, subset_idcs[i]) for i in range(n_clients)]

  client_loaders = [torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True) for subset in client_data]
  test_loader = torch.utils.data.DataLoader(test_data, batch_size=100)

  return client_loaders, test_loader, client_data, test_data



def split_image_data(labels, n_clients, n_data, classes_per_client):

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



def split_dirichlet(labels, n_clients, n_data, alpha, double_stochstic=True):
    '''Splits data among the clients according to a dirichlet distribution with parameter alpha'''
    n_classes = labels[0].size(0)
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)

    if double_stochstic:
      label_distribution = make_double_stochstic(label_distribution)

    class_idcs = [np.argwhere(torch.argmax(torch.stack(labels), dim=1).detach().numpy()==y).flatten() 
           for y in range(n_classes)]

    client_idcs = [[] for _ in range(n_clients)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    print_split(client_idcs, labels)
  
    return client_idcs


def make_double_stochstic(x):
    rsum = None
    csum = None

    n = 0 
    while n < 1000 and (np.any(rsum != 1) or np.any(csum != 1)):
        x /= x.sum(0)
        x = x / x.sum(1)[:, np.newaxis]
        rsum = x.sum(1)
        csum = x.sum(0)
        n += 1

    #x = x / x.sum(axis=0).reshape(1,-1)
    return x



def print_split(idcs, labels):
  n_labels = labels[0].size(0) 
  print("Data split:")
  splits = []
  for i, idccs in enumerate(idcs):
    split = np.sum(torch.argmax(torch.stack(labels), dim=1).detach().numpy()[idccs].reshape(1,-1)==np.arange(n_labels).reshape(-1,1), axis=1)
    splits += [split]
    if i < 10 or i>len(idcs)-10:
      print(" - Client {}: {:55} -> sum={}".format(i,str(split), np.sum(split)), flush=True)
    elif i==len(idcs)-10:
      print(".  "*10+"\n"+".  "*10+"\n"+".  "*10)

  print(" - Total:      {}".format(np.stack(splits, axis=0).sum(axis=0)))
  print()

class DataloaderMerger(object):
  def __init__(self, loaders):
    assert isinstance(loaders, dict), 'Please initialize the DataloaderMerger with a dict of names and dataloaders'
    self.loaders = loaders

  def __setitem__(self, key, value):
    self.loaders[key] = value

  def __iter__(self):
    self.iters = [iter(loader) for loader in self.loaders.values()]
    self.iters[1:] = [cycle(loader) for loader in self.iters[1:]]
    return self

  def __next__(self):
    try:
      x,y,source = None, None, None
      if len(self.iters) > 1:
        x,y = zip(*(next(iterator) for iterator in self.iters))
        source = torch.cat([i*torch.ones(subbatch.size(0)) for i,subbatch in enumerate(x)] )
        x = torch.cat(x)
        y = torch.cat(y)
      else:
        x,y = next(self.iters[0])
        source = torch.zeros(x.size(0))
      return x,y, source
    except:
        raise StopIteration


