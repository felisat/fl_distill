import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class VGG(nn.Module):

    def __init__(self, cfg, size=512, out=10):
        super(VGG, self).__init__()

        self.features = self.make_layers(cfg)
        self.classifier = nn.Sequential(
            #nn.Dropout(),
            nn.Linear(size, size),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(size, size),
            nn.ReLU(True),
            nn.Linear(size, out),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

def vgg11s():
    return VGG([32, 'M', 64, 'M', 128, 128, 'M', 128, 128, 'M', 128, 128, 'M'], size=128)
  
def vgg16():
    return VGG([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'])



class lenet_cifar(nn.Module):
    def __init__(self):
        super(lenet_cifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



def get_model(model):

  return  { "vgg16" : (vgg16, optim.SGD, {"lr":0.04, "momentum":0.9, "weight_decay":5e-5}),
            "vgg11s" : (vgg11s, optim.SGD, {"lr":0.04, "momentum":0.9, "weight_decay":5e-5}),
              "lenet_cifar" : (lenet_cifar, optim.SGD, {"lr":0.002, "momentum":0.9, "weight_decay":5e-5})
          }[model]


def print_model(model):
  n = 0
  print("Model:")
  for key, value in model.named_parameters():
    print(' -', '{:30}'.format(key), list(value.shape))
    n += value.numel()
  print("Total number of Parameters: ", n) 
  print()












