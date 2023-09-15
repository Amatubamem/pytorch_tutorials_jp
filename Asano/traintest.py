'''
学習およびテスト
損失関数が(x,y)の2変数関数でないような場合，model.lossNNに定義して処理する
testはクラス分類タスクを前提にしているがそれ以外でもAccuracyが0で出てくるだけ
'''



import torch
from torch import nn
import models

device = torch.device('mps' if torch.backends.mps.is_available() else "cpu") 

def train(dataloader, model, optimizer, loss_fn):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # print(nn.Flatten(X))

        X, y = X.to(device), y.to(device)

        # 損失誤差を計算
        pred = model(X)
        loss = loss_fn(pred, y)

        # 損失関数の受け取る引数がモデルによって異なるがmodels側で対応
        # if type(model) == models.Classifier:
        #     loss = loss_fn(pred, y)
        # elif type(model) == models.AutoEncoder:
        #     loss = loss_fn(pred, nn.Flatten()(X))
        # elif type(model) == models.SparseAutoEncoder:
        #     loss = loss_fn(pred, nn.Flatten()(X))
        
        # バックプロパゲーション
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn): # for classification
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()

            if type(model) == models.Classifier:
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct, test_loss