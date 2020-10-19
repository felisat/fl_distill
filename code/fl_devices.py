import random
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np 
from models import outlier_net

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

    self.c_round = 0
    
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
  def __init__(self, model_fn, optimizer_fn, loader, init=None, idnum=None):
    super().__init__(model_fn, optimizer_fn, loader, init)
    self.id = idnum
    self.feature_extractor = None
    
  def synchronize_with_server(self, server, c_round):
    self.model.load_state_dict(server.model.state_dict())
    self.c_round = c_round
    #copy(target=self.W, source=server.W)
    
  def compute_weight_update(self, epochs=1, loader=None, reset_optimizer=True):
    if reset_optimizer:
      self.optimizer = self.optimizer_fn(self.model.parameters())  
    train_stats = train_op(self.model, self.loader if not loader else loader, self.optimizer, self.scheduler, epochs)
    return train_stats

  def predict(self, x):
    """Softmax prediction on input"""
    self.model.eval()
    with torch.no_grad():
      y_ = nn.Softmax(1)(self.model(x))

    return y_

  def predict_logit(self, x):
    """Softmax prediction on input"""
    self.model.eval()
    with torch.no_grad():
      y_ = self.model(x)

    return y_


  def predict_(self, x):
    """Argmax prediction on input"""
    self.model.eval()
    with torch.no_grad():
      y_ = nn.Softmax(1)(self.model(x))
      
      amax = torch.argmax(y_, dim=1).detach()#(torch.cumsum(y_, dim=1)<torch.rand(size=(y_.shape[0],1))).sum(dim=1)

      t = torch.zeros_like(y_)
      t[torch.arange(y_.shape[0]),amax] = 1

    return t.detach()


  def predict_random(self, x):
    """Softmax prediction on input"""
    adv = torch.zeros(size=[x.shape[0],10], device="cuda")
    adv[:,0] = 1.0
    #rand = torch.rand(size=[x.shape[0],10]).cuda()
    #rand = rand / torch.sum(rand, axis=-1).reshape(-1,1)
    return adv


  def generate_feature_bank(self):
    """Extracts Features of local data using local neural network"""
    feature_bank, feature_labels = [], []
    with torch.no_grad():
        # generate feature bank
        for data, target in self.loader:
            feature = self.model.f(data.to(device)).squeeze() 
            #feature = feature / torch.norm(feature, dim=1).view(-1,1)
            feature_bank.append(feature)
            feature_labels.append(target)
        # [D, N]
        self.feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        self.feature_labels = torch.cat(feature_labels)
        # loop test data to predict the label by weighted knn search

  def predict_knn_weight(self, x, k=10):
    """Weighted k-neirest-neighbor-prediction in feature space (first a feature bank needs to be generated)"""
    with torch.no_grad():
      x = x.to(device)
      feature = self.model.f(x).squeeze()
      #feature = feature / torch.norm(feature, dim=1).view(-1,1)

      # compute cos similarity between each feature vector and feature bank ---> [B, N]
      sim_matrix = torch.mm(feature, self.feature_bank)
      # [B, K]
      sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)

    return torch.mean(sim_weight,axis=-1)


  def predict_knn(self, x, k=10, n_classes=10):
    """Weighted k-neirest-neighbor-prediction in feature space (first a feature bank needs to be generated)"""
    with torch.no_grad():
      x = x.to(device)
      feature = self.model.f(x).squeeze()
      feature = feature / torch.norm(feature, dim=1).view(-1,1)

      # compute cos similarity between each feature vector and feature bank ---> [B, N]
      sim_matrix = torch.mm(feature, self.feature_bank)
      # [B, K]
      sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
      # [B, K]
      sim_labels = torch.gather(self.feature_labels.expand(x.size(0), -1).to(device), dim=-1, index=sim_indices.to(device))
      #sim_weight = (sim_weight / temperature).exp()

      # counts for each class
      one_hot_label = torch.zeros(x.size(0) * k, n_classes, device=sim_labels.device)
      # [B*K, C]
      one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.long().view(-1, 1), value=1.0)
      # weighted score ---> [B, C]
      pred_scores = torch.sum(one_hot_label.view(x.size(0), -1, n_classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    return pred_scores.detach()



  def compute_prediction_matrix(self, distill_loader):

    predictions = []
    for x, _ in distill_loader:
      x = x.to(device)
      y_predict = self.predict(x).detach()
      
      predictions += [y_predict]

    return torch.cat(predictions, dim=0).detach().cpu().numpy()


  def train_outlier_detector(self):
    self.outlier_model = outlier_net().cuda()
    optimizer = torch.optim.Adam(self.outlier_model.parameters(), lr=0.001, weight_decay=1e-6)

    x = iter(self.loader).next()[0].cuda()
    self.outlier_c = torch.mean(self.outlier_model(x), dim=0).detach()

    n_epochs = 100
    for epoch in range(n_epochs+1):
        train_loss = 0.0
        for x, _ in self.loader:
            x = x.cuda()
            optimizer.zero_grad()
            r = self.outlier_model(x)
            loss = torch.mean((r -self.outlier_c)**2)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*x.size(0)
                
        train_loss = train_loss/len(self.loader)
    print('Epoch: {} \tTraining Loss: {}'.format(epoch, train_loss))
    self.mean_outlier_loss = train_loss


  def predict_outlier_score(self, x):
    with torch.no_grad():
      r = self.outlier_model(x)
      loss = torch.mean((r - self.outlier_c)**2, dim=-1)

    return loss.detach()/self.mean_outlier_loss


  def train_outlier_detector(self, model, distill_loader, **kw_args):
    from sklearn.decomposition import PCA
    from sklearn.svm import OneClassSVM
    from sklearn.ensemble import IsolationForest
    from sklearn.neural_network import MLPRegressor
    from  sklearn.neighbors import KernelDensity


    if self.feature_extractor is not None:
      X_train = torch.cat([self.feature_extractor.f(x[0][:,:].cuda()).detach().cpu() for x in self.loader], dim=0).reshape(-1,512).numpy()
    else:
      X_train = torch.cat([x[0][:,0] for x in self.loader], dim=0).reshape(-1,1*32*32).numpy()
      X_distill = torch.cat([x[0][0][:,0] for x in distill_loader], dim=0).reshape(-1,1*32*32).numpy()

    pca = PCA(n_components=0.95, svd_solver="full", whiten=True)
    pca.fit(X_train)
    X_train_ = pca.transform(X_train)
    X_distill_ = pca.transform(X_distill)



    if model=="ocsvm":
      self.outlier_model = OneClassSVM(**kw_args).fit(X_train_)
      self.outlier_model.score = lambda x : self.outlier_model.score_samples(x)
    
    elif model=="isolation_forest":
      self.outlier_model = IsolationForest(**kw_args).fit(X_train_)

    elif model=="kde":
      self.outlier_model = KernelDensity(**kw_args).fit(X_train_)
      self.outlier_model.score = lambda x : np.exp(self.outlier_model.score_samples(x))

    elif model=="autoencoder":
      self.outlier_model = MLPRegressor(hidden_layer_sizes = (200, 100, 50, 100, 200), 
                   activation = 'relu', 
                   solver = 'adam', 
                   learning_rate_init = 0.0001, 
                   max_iter = 100, 
                   verbose = False,
                   **kwargs).fit(X_train_, X_train_)

    scores = self.outlier_model.score(X_distill_)

    norm_scores = (scores-np.min(scores))/(np.max(scores)-np.min(scores))
    #norm_scores = sigmoid((scores - np.mean(scores))/np.std(scores))
    #self.outlier_model.pca = pca
    #self.outlier_model.mean = np.mean()
    #self.outlier_model.std = np.std(self.outlier_model.score_samples(X_distill_))

    self.scores = torch.Tensor(norm_scores)

  def predict_outlier_score(self, idcs):
    return self.scores[idcs]


"""  def predict_outlier_score(self, x):

    if self.feature_extractor is not None:
      x = self.feature_extractor.f(x.cuda()).detach().cpu().numpy().reshape(-1,512)
    else:
      x = x[:,0].cpu().detach().numpy().reshape(-1,1*32*32)
    x = self.outlier_model.pca.transform(x)

    return self.outlier_model.score_samples(x)#sigmoid((self.outlier_model.score_samples(x)-self.outlier_model.mean)/self.outlier_model.std)#"""




    
 
class Server(Device):
  def __init__(self, model_fn, optimizer_fn, loader, unlabeled_loader, init=None):
    super().__init__(model_fn, optimizer_fn, loader, init)
    self.distill_loader = unlabeled_loader
    
  def select_clients(self, clients, frac=1.0):
    return random.sample(clients, int(len(clients)*frac)) 
    
  def aggregate_weight_updates(self, clients):
    reduce_average(target=self.W, sources=[client.W for client in clients])


  def distill(self, clients, epochs=1, mode="regular"):
    print("Distilling...")
    self.model.train()  
    #vat_loss = VATLoss(xi=10.0, eps=1.0, ip=1)

    valid_modes = ["regular", "mean_logits", "weighted", "pate", "knn", "pate_up", "momentum", "count_weighted", "outlier_score", "adversaries"]
    assert mode in valid_modes, "mode has to be one of {}".format(valid_modes)

    # Use only clients who participated in the previous round for distilling except if momentum mode
    c_max = max([client.c_round for client in clients])
    if mode not in ["momentum"]: 
      clients = [c for c in clients if c.c_round==c_max] 

    acc = 0
    for ep in range(epochs):
      running_loss, samples = 0.0, 0
      for (x, _), idx in tqdm(self.distill_loader):   
        x = x.to(device)

        if mode == "momentum":
          y = torch.zeros([x.shape[0], 10], device="cuda")
          w = 0
          
          for i, client in enumerate(clients):
            if client.c_round > 0:
              y_p = client.predict(x)
              y += 0.8**(c_max - client.c_round)*y_p.detach()
              w += 0.8**(c_max - client.c_round)
          y = y / w

        if mode == "count_weighted":
          y = torch.zeros([x.shape[0], 10], device="cuda")
          for i, client in enumerate(clients):
            normalized_counts = torch.Tensor(client.label_counts/client.label_counts.sum()).cuda()

            y_p = 10**(client.predict(x)-normalized_counts)
            y_p /= y_p.sum(dim=1).view(-1,1)

            y += (y_p/len(clients)).detach()        


        if mode == "regular":
          y = torch.zeros([x.shape[0], 10], device="cuda")
          for i, client in enumerate(clients):
            y_p = client.predict(x)
            y += (y_p/len(clients)).detach()

        if mode == "adversaries":
          y = torch.zeros([x.shape[0], 10], device="cuda")
          for i, client in enumerate(clients):
            if client.is_adversary:
              y_p = client.predict_random(x)
            else:
              y_p = client.predict(x)
            y += (y_p/len(clients)).detach()


        if mode == "mean_logits":
          y = torch.zeros([x.shape[0], 10], device="cuda")
          for i, client in enumerate(clients):
            y_p = client.predict_logit(x)
            y += (y_p/len(clients)).detach()

          y = nn.Softmax(1)(y)

        if mode == "weighted":
          y = torch.zeros([x.shape[0], 10], device="cuda")
          w = torch.zeros([x.shape[0], 1], device="cuda")
          for i, client in enumerate(clients):
            y_p = client.predict(x)
            weight = client.predict_knn_weight(x).view(-1,1)

            y += (y_p*weight).detach()
            w += weight.detach()
          y = y / w


        if mode == "outlier_score":
          y = torch.zeros([x.shape[0], 10], device="cuda")
          w = torch.zeros([x.shape[0], 1], device="cuda")
          for i, client in enumerate(clients):
            y_p = client.predict(x)
            weight = client.predict_outlier_score(idx).reshape(-1,1).cuda()

            #print(client.label_counts)
            #for i,j in zip(_,weight):
            #  print(i, j)
            #exit()

            y += (y_p*weight).detach()
            w += weight.detach()
          y = y / w


        if mode == "pate":
          hist = torch.sum(torch.stack([client.predict_(x) for client in clients]), dim=0)
          #hist += torch.randn_like(hist)

          amax = torch.argmax(hist, dim=1)

          y = torch.zeros_like(hist)
          y[torch.arange(hist.shape[0]),amax] = 1


        if mode == "pate_up":
          y = torch.mean(torch.stack([client.predict_(x) for client in clients]), dim=0)


        if mode in "knn":
          scores = torch.zeros([x.shape[0], 10], device="cuda")
          for i, client in enumerate(clients):
            y_p = client.predict_knn(x, k=10)
            scores += (y_p/len(clients)).detach()          

          amax = torch.argmax(scores, dim=1)

          y = torch.zeros_like(scores)
          y[torch.arange(x.shape[0]),amax] = 1

        self.optimizer.zero_grad()

        #vat = vat_loss(model, x)

        y_ = nn.Softmax(1)(self.model(x))

        def kulbach_leibler_divergence(predicted, target):
          return -(target * torch.log(predicted.clamp_min(1e-7))).sum(dim=-1).mean() - \
                 -1*(target.clamp(min=1e-7) * torch.log(target.clamp(min=1e-7))).sum(dim=-1).mean()

        loss = kulbach_leibler_divergence(y_,y)#torch.mean(torch.sum(y_*(y_.log()-y.log()), dim=1))

 
        running_loss += loss.item()*y.shape[0]
        samples += y.shape[0]

        loss.backward()
        self.optimizer.step()  

      #print(running_loss/samples)


      acc_new = eval_op(self.model, self.loader)["accuracy"]
      print(acc_new)

      if acc_new < acc:
        return {"loss" : running_loss / samples, "acc" : acc_new, "epochs" : ep}
      else:
        acc = acc_new

    return {"loss" : running_loss / samples, "acc" : acc_new, "epochs" : ep}



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



def eval_op(model, loader):
    model.train()
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



def sigmoid(x):
  return np.exp(x)/(1+np.exp(x))