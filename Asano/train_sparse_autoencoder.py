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


input_dim = 28**2
output_dim = 1000
index = 104

import models
model = models.SparseAutoEncoder(dimx=input_dim, dimy=output_dim)

prefix = f'AE{output_dim}-{index}'
filename = prefix + '.pth'
if os.path.isfile(filename):
    model.load_state_dict(torch.load(filename))


model = model.to(device)
print(model)



learning_rate = 1e-3 #hyperparameter

import lossfunc as lf
loss_fn = model.lossNN
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


epochs = 50

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

    
print("Done!")

fig = vis.visualizer(e, c, l) 
fig.savefig(f"{prefix}.png") 



torch.save(model.state_dict(), filename)
print(f"saved to {filename}")


# 計測終了
end_time = time.time()

# 実行時間を計算
execution_time = end_time - start_time

# 結果を表示
print(f"プログラムの実行時間: {execution_time}秒")