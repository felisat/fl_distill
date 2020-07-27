import random
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn



device = 'cuda' if torch.cuda.is_available() else 'cpu'
    

class Device(object):
  def __init__(self, model_fn, optimizer_fn, loader, init=None):
    self.model = model_fn().to(device)
    self.loader = loader

    self.W = {key : value for key, value in self.model.named_parameters()}
    self.dW = {key : torch.zeros_like(value) for key, value in self.model.named_parameters()}
    self.W_old = {key : torch.zeros_like(value) for key, value in self.model.named_parameters()}

    self.optimizer = optimizer_fn(self.model.parameters())     
    
  def evaluate(self, loader=None):
    return eval_op(self.model, self.loader if not loader else loader)

  def save_model(self, path=None, name=None, verbose=True):
    if name:
      torch.save(self.model.state_dict(), path+name)
      if verbose: print("Saved model to", path+name)

  def load_model(self, path=None, name=None, verbose=True):
    if name:
      self.model.load_state_dict(torch.load(path+name))
      if verbose: print("Loaded model from", path+name)
    
      
class Client(Device):
  def __init__(self, model_fn, optimizer_fn, loader, init=None):
    super().__init__(model_fn, optimizer_fn, loader, init)
    
  def synchronize_with_server(self, server):
    copy(target=self.W, source=server.W)
    
  def compute_weight_update(self, epochs=1, loader=None):
    train_stats = train_op(self.model, self.loader if not loader else loader, self.optimizer, epochs)
    return train_stats

  def predict(self, x):
    return nn.Softmax(1)(self.model(x)) 

    
 
class Server(Device):
  def __init__(self, model_fn, optimizer_fn, loader, unlabeled_loader, init=None):
    super().__init__(model_fn, optimizer_fn, loader, init)
    self.distill_loader = unlabeled_loader
    
  def select_clients(self, clients, frac=1.0):
    return random.sample(clients, int(len(clients)*frac)) 
    
  def aggregate_weight_updates(self, clients):
    reduce_average(target=self.W, sources=[client.W for client in clients])


  def distill(self, clients, epochs=1, loader=None):
    print("Distilling...")
    distill_op(self.model, [client.model for client in clients], self.distill_loader if not loader else loader, self.optimizer, epochs)

    


def train_op(model, loader, optimizer, epochs):
    model.train()  
    running_loss, samples = 0.0, 0
    for ep in range(epochs):
      for x, y in loader:   
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()

        loss = nn.CrossEntropyLoss()(model(x), y)
        running_loss += loss.item()*y.shape[0]
        samples += y.shape[0]

        loss.backward()
        optimizer.step()  

    return {"loss" : running_loss / samples}


def kulbach_leibler_divergence(predicted, target):
    return -(target * torch.log(predicted.clamp_min(1e-7))).sum(dim=-1).mean() - \
           -1*(target.clamp(min=1e-7) * torch.log(target.clamp(min=1e-7))).sum(dim=-1).mean()

def distill_op(model, client_models, loader, optimizer, epochs):
    model.train()  
    running_loss, samples = 0.0, 0
    for ep in range(epochs):
      for x, _ in tqdm(loader):   
        x = x.to(device)

        y = torch.mean(torch.stack([nn.Softmax(1)(m(x)) for m in client_models]), dim=0)

        optimizer.zero_grad()

        y_ = nn.Softmax(1)(model(x))

        loss = kulbach_leibler_divergence(y_,y)#torch.mean(torch.sum(y_*(y_.log()-y.log()), dim=1))
 
        running_loss += loss.item()*y.shape[0]
        samples += y.shape[0]

        loss.backward()
        optimizer.step()  

    return {"loss" : running_loss / samples}


def eval_op(model, loader):
    model.train()
    samples, correct = 0, 0

    with torch.no_grad():
      for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        y_ = model(x)
        _, predicted = torch.max(y_.data, 1)
        
        samples += y.shape[0]
        correct += (predicted == y).sum().item()

    return {"accuracy" : correct/samples}



def flatten(source):
  return torch.cat([value.flatten() for value in source.values()])

def copy(target, source):
  for name in target:
    target[name].data = source[name].data.clone()
    
def reduce_average(target, sources):
  for name in target:
      target[name].data = torch.mean(torch.stack([source[name].data for source in sources]), dim=0).clone()

def subtract_(target, minuend, subtrahend):
  for name in target:
    target[name].data = minuend[name].data.clone()-subtrahend[name].data.clone()



