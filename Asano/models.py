'''
作成したネットワークモデルはここに定義
使用する損失関数が(pred, y)の2変数関数でない場合にはここでlossNNとして定義する．
使いたい損失関数のバリエーションが複数ある場合はどうしようね，と思ったんだけど普通にメソッド追加すればいいね，
SparseAutoEncoderは消去してAutoEncoderに新しい損失関数を定義しようか．
'''

from torch import nn
import numpy as np
import lossfunc as lf


class AutoEncoder(nn.Module):
    def __init__(self, dimx=28**2, dimy=512): #hyperparameter
        super(AutoEncoder, self).__init__()
        self.flatten = nn.Flatten()
        self.encoder = nn.Sequential(
            nn.Linear(dimx, dimy),
            nn.ReLU()
        )
        self.decoder = nn.Linear(dimy, dimx)

    def forward(self, x):
        x = self.flatten(x)
        self.code = self.encoder(x)
        decoded = self.decoder(self.code)
        return decoded
    
    def lossNN(self,x,y):
        loss_val = nn.MSELoss(reduction='mean')(self.forward(x), nn.Flatten()(x))
        return loss_val
    

class SparseAutoEncoder(AutoEncoder):
    def __init__(self, *args, **kwargs): #hyperparameter
        super(SparseAutoEncoder, self).__init__(*args, **kwargs)
    
    def lossNN(self,x,y):
        loss_val = lf.SparseMSELoss(reduction='mean')(self.forward(x), nn.Flatten()(x), self.code)
        return loss_val

class Classifier(nn.Module):
    def __init__(self, layers=[28**2, 512, 512, 10], activation=nn.ReLU(), autoencoder=None):
        super(Classifier, self).__init__()
        self.flatten = nn.Flatten()
        self.encoder = autoencoder.encoder if autoencoder else None

        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])


        self.linear_relu_stack = nn.Sequential(
            *[
                activation if i%2 else self.linears[i//2] for i in range((len(layers)-1) * 2 -1)
            ],
            nn.Softmax()
        )

    def forward(self, x):
        x = self.flatten(x)
        if self.encoder:
            x = self.encoder(x)
        logits = self.linear_relu_stack(x)
        return logits
    
    def lossNN(self,x,y):
        loss_val = nn.CrossEntropyLoss()(x, y)
        return loss_val

