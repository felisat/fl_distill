import random
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn

from virtual_adversarial_training import VATLoss



device = 'cuda' if torch.cuda.is_available() else 'cpu'
    

class Device(object):
  def __init__(self, model_fn, optimizer_fn, loader, init=None):
    self.model = model_fn().to(device)
    self.loader = loader

    self.W = {key : value for key, value in self.model.named_parameters()}
    self.dW = {key : torch.zeros_like(value) for key, value in self.model.named_parameters()}
    self.W_old = {key : torch.zeros_like(value) for key, value in self.model.named_parameters()}

    self.optimizer_fn = optimizer_fn
    self.optimizer = optimizer_fn(self.model.parameters())   
    self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.96)  
    
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
    train_stats = train_op(self.model, self.loader if not loader else loader, self.optimizer, self.scheduler, epochs)
    return train_stats

  def predict(self, x):
    y_ = nn.Softmax(1)(self.model(x))

    return y_


  def predict_(self, x):
    with torch.no_grad():
      y_ = nn.Softmax(1)(self.model(x))
      
      amax = torch.argmax(y_, dim=1).detach()#(torch.cumsum(y_, dim=1)<torch.rand(size=(y_.shape[0],1))).sum(dim=1)

      t = torch.zeros_like(y_)
      t[torch.arange(y_.shape[0]),amax] = 1

    return t.detach()




    
 
class Server(Device):
  def __init__(self, model_fn, optimizer_fn, loader, unlabeled_loader, init=None):
    super().__init__(model_fn, optimizer_fn, loader, init)
    self.distill_loader = unlabeled_loader
    
  def select_clients(self, clients, frac=1.0):
    return random.sample(clients, int(len(clients)*frac)) 
    
  def aggregate_weight_updates(self, clients):
    reduce_average(target=self.W, sources=[client.W for client in clients])


  def distill(self, clients, epochs=1, loader=None, eval_loader=None, compress=False, noise=False):
    print("Distilling...")
    return distill_op(self.model, clients, self.distill_loader if not loader else loader, self.loader if not eval_loader else eval_loader, self.optimizer, epochs, compress=compress, noise=noise)

  #def recalibrate_batchnorm(self, loader=None):
  #  compute_batchnorm_running_statistics(self.model, self.distill_loader if not loader else loader)

    

#def compute_batchnorm_running_statistics(model, loader):
#    model.train()
#    for ep in tqdm(range(20)):
#      for x, y in loader:
#        x, y = x.to(device), y.to(device)
#        model(x)  


def train_op(model, loader, optimizer, scheduler, epochs):
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
      #scheduler.step()

    return {"loss" : running_loss / samples}


def kulbach_leibler_divergence(predicted, target):
    return -(target * torch.log(predicted.clamp_min(1e-7))).sum(dim=-1).mean() - \
           -1*(target.clamp(min=1e-7) * torch.log(target.clamp(min=1e-7))).sum(dim=-1).mean()


def compress_soft_labels(y_):
      sample = (torch.cumsum(y_, dim=1)<torch.rand(size=(y_.shape[0],1)).to(device)).sum(dim=1)

      t = torch.zeros_like(y_).to(device)
      t[torch.arange(y_.shape[0]),sample] = 1

      return t


def distill_op(model, clients, loader, eval_loader, optimizer, epochs, compress=False, noise=False):
    model.train()  

    #vat_loss = VATLoss(xi=10.0, eps=1.0, ip=1)

    acc = 0
    import time
    for ep in range(epochs):
      running_loss, samples = 0.0, 0
      for x, _ in tqdm(loader):   
        x = x.to(device)

        if not noise:


          y = torch.zeros([x.shape[0], 10], device="cuda")
          for i, client in enumerate(clients):
            y_p = client.predict(x)
            y += (y_p/len(clients)).detach()

          #y = torch.mean(torch.stack([client.predict(x) for client in clients]), dim=0)
        else:
          hist = torch.sum(torch.stack([client.predict_(x) for client in clients]), dim=0)
          hist += torch.randn_like(hist)

          amax = torch.argmax(hist, dim=1)

          y = torch.zeros_like(hist)
          y[torch.arange(hist.shape[0]),amax] = 1



        optimizer.zero_grad()

        #vat = vat_loss(model, x)

        y_ = nn.Softmax(1)(model(x))

        loss = kulbach_leibler_divergence(y_,y)#torch.mean(torch.sum(y_*(y_.log()-y.log()), dim=1))

 
        running_loss += loss.item()*y.shape[0]
        samples += y.shape[0]

        loss.backward()
        optimizer.step()  

      #print(running_loss/samples)


      acc_new = eval_op(model, eval_loader)["accuracy"]
      print(acc_new)

      if acc_new < acc:
        return {"loss" : running_loss / samples, "acc" : acc_new, "epochs" : ep}
      else:
        acc = acc_new

    return {"loss" : running_loss / samples, "acc" : acc_new, "epochs" : ep}


def eval_op(model, loader):
    model.eval()
    samples, correct = 0, 0

    with torch.no_grad():
      for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        y_ = model(x)
        _, predicted = torch.max(y_.detach(), 1)
        
        samples += y.shape[0]
        correct += (predicted == y).sum().item()

    return {"accuracy" : correct/samples}



def flatten(source):
  return torch.cat([value.flatten() for value in source.values()])

def copy(target, source):
  for name in target:
    target[name].data = source[name].detach().clone()
    
def reduce_average(target, sources):
  for name in target:
      target[name].data = torch.mean(torch.stack([source[name].detach() for source in sources]), dim=0).clone()

def subtract_(target, minuend, subtrahend):
  for name in target:
    target[name].data = minuend[name].detach().clone()-subtrahend[name].detach().clone()



