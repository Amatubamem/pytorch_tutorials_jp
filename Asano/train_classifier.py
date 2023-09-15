import numpy as np
from numpy import sin, cos, pi, floor, exp, log, log10, sqrt, cbrt
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import os
import time
import random


import models
import traintest as tt
import visualizer as vis




start_time = time.time()


# load datasets
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# データローダーの作成 
batch_size = 64 #hyperparameter

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break
# device setting
device = torch.device('mps' if torch.backends.mps.is_available() else "cpu") 
print("Using {} device".format(device))


ae = None
input_dim = 28**2
AE_dim = 512
output_dim = 10
layers = [AE_dim, 256, output_dim]

index = 'test'

prefix = f'CL{len(layers)}l(' + ','.join([str(n) for n in layers[:-1]]) + f'){index}'


AEfilename = ['', f'AE{AE_dim}-{100}.pth'][0]
filename = prefix + '.pth'

if os.path.isfile(AEfilename):
    ae = models.AutoEncoder(dimx=28**2, dimy=AE_dim).to(device)
    ae.load_state_dict(torch.load(AEfilename))

    if os.path.isfile(filename):
        model = models.Classifier(layers=layers, autoencoder=ae)
        model.load_state_dict(torch.load(filename))
    else:
        model = models.Classifier(layers=layers, autoencoder=ae)

else:
    if os.path.isfile(filename):
        model = models.Classifier(layers=[input_dim, *layers], autoencoder=ae)
        model.load_state_dict(torch.load(filename))
    else:
        model = models.Classifier(layers=[input_dim, *layers], autoencoder=ae)


model = model.to(device)
print(model)



learning_rate = 1e-3 #hyperparameter
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

pre_epochs = 0
epochs = 10

e = []
c = []
l = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    tt.train(train_dataloader, model, optimizer, loss_fn) 
    correct, test_loss = tt.test(test_dataloader, model, loss_fn)
    e.append(t)
    c.append(correct)
    l.append(test_loss)
torch.save(model.state_dict(), filename)
print(f"saved to {filename}")

    
print("Done!")

fig = vis.visualizer(e, c, l) 
fig.savefig(f"{prefix}({pre_epochs}~{pre_epochs+epochs}){AEfilename[:-4]}.png") 


end_time = time.time()

print(f'実行時間:{end_time-start_time}')