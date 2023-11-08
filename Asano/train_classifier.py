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
import csv


import models
import traintest as tt
import visualizer as vis


# parameter setting
epochs = 1000

input_dim = 28**2
output_dim = 10


batch_size = 64
AE_dim = 1000
layers = [AE_dim, 256, output_dim]

index = 'L02'

useAE = True
AEindex = '104'

learning_rate = 1e-3 #hyperparameter

pre_epochs = 0

prefix = f'CL{len(layers)}l(' + ','.join([str(n) for n in layers[:-1]]) + f'){index}'
filename = prefix + '.pth'

AEfilename = '' if not useAE else f'AE{AE_dim}-{AEindex}.pth'

accuracy = []
aveloss = []

if os.path.isfile(prefix + '.csv'):
    parameters = {}
    with open(prefix+'.csv', 'r') as settings:
        settings_reader = csv.DictReader(settings)
        for row in settings_reader:
            pname = row['Name']
            pvalue = row['Value']

            if ',' in pvalue:
                pvalue = pvalue.split(',')
            parameters[pname] = pvalue

    input_dim = int(parameters['input_dim'])
    output_dim = int(parameters['output_dim'])

    batch_size = int(parameters['batch_size'])
    AE_dim = int(parameters['AE_dim'])
    layers = list(map(int,parameters['layers']))
    index = parameters['index']
    useAE = True if parameters['useAE'] else False
    AEindex = parameters['AEindex']
    learning_rate = float(parameters['learning_rate'])
    accuracy = list(map(float, parameters['accuracy']))
    aveloss = list(map(float, parameters['aveloss']))


    pre_epochs = len(accuracy)




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
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break
# device setting
device = torch.device('mps' if torch.backends.mps.is_available() else "cpu") 
print("Using {} device".format(device))

# 初期パラメータの設定

if os.path.isfile(AEfilename) and useAE:
    ae = models.AutoEncoder(dimx=28**2, dimy=AE_dim).to(device)
    ae.load_state_dict(torch.load(AEfilename))

    if os.path.isfile(filename):
        model = models.Classifier(layers=layers, autoencoder=ae)
        model.load_state_dict(torch.load(filename))
    else:
        model = models.Classifier(layers=layers, autoencoder=ae)

else:
    if os.path.isfile(filename):
        model = models.Classifier(layers=[input_dim, *layers])
        model.load_state_dict(torch.load(filename))
    else:
        model = models.Classifier(layers=[input_dim, *layers])


model = model.to(device)
print(model)



optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()


for t in range(epochs):
    print(f"Epoch {pre_epochs+t+1}\n-------------------------------")
    tt.train(train_dataloader, model, optimizer, loss_fn) 
    correct, test_loss = tt.test(test_dataloader, model, loss_fn)
    accuracy.append(correct)
    aveloss.append(test_loss)
torch.save(model.state_dict(), filename)
print(f"saved to {filename}")

    
print("Done!")

fig = vis.progplot(epochs+pre_epochs, accuracy, aveloss) 
fig.savefig(f"{prefix}({pre_epochs}~{pre_epochs+epochs}){AEfilename[:-4]}.png") 


parameters = {
    'input_dim': input_dim,
    'output_dim': output_dim,
    'batch_size': batch_size,
    'AE_dim': AE_dim,
    'layers': layers,
    'index': index,
    'useAE': int(useAE),
    'AEindex': AEindex,
    'learning_rate': learning_rate,
    'accuracy': accuracy,
    'aveloss': aveloss
}

with open(prefix+'.csv', 'w', newline='') as settings:
    fieldnames = ['Name', 'Value']
    csv_writer = csv.DictWriter(settings, fieldnames=fieldnames)
    
    # CSVファイルのヘッダ行を書き込む
    csv_writer.writeheader()
    
    # パラメータを辞書から取得し、CSVファイルに書き込む
    for pname, pvalue in parameters.items():
        # リストの場合、カンマで区切った文字列に変換
        if isinstance(pvalue, list):
            pvalue = ','.join(map(str,pvalue))
        
        csv_writer.writerow({'Name': pname, 'Value': pvalue})

end_time = time.time()

print(f'実行時間:{end_time-start_time}')