import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch
import os
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_predict
from sklearn import preprocessing
from skorch.regressor import NeuralNetRegressor
from sklearn.model_selection import KFold

df = pd.read_csv('Property_target.csv')
df = df.drop(columns = ['Unnamed: 0'])
prop = df['thermal'] # Target properties

tag = 'TL'


desc_proxy = pd.read_csv(
	'proxy_desc_agl_mp.csv') # Proxy desc  
desc_proxy = desc_proxy.drop(columns = ['Unnamed: 0'])


desc_target = pd.read_csv(
	'target_desc.csv') # Target desc  
desc_target = desc_target.drop(columns = ['Unnamed: 0'])



scaler = preprocessing.MinMaxScaler()
desc_proxy_scaler = scaler.fit_transform(desc_proxy)
desc_target_scaler = scaler.transform(desc_target)

parameter_list = np.loadtxt('parameter_list')
cwd = os.getcwd()


device = 'cuda'



class TransNet(nn.Module):
    def __init__(self):
        super(TransNet, self).__init__()
        self.sharedlayer = nn.Sequential(
            nn.Linear(feature_size, h_layer1),
            nn.ReLU(),
            nn.Linear(h_layer1, h_layer2),
            nn.ReLU(),
            nn.Linear(h_layer2, h_layer3),
            nn.ReLU(),
        )
        self.finallayer = nn.Sequential(
            nn.Linear(h_layer3, out_layer)
        )
    def forward(self, x):
        x = self.sharedlayer(x)
        h_shared  = self.finallayer(x)
        return h_shared


device = 'cuda'
LR = 0.001
MAX_EPOCHS = 500

feature_size = desc_target.shape[1]
out_layer = 1
LR = 0.001
BATCH_SIZE = int(len(df)/2)

for i in range(len(parameter_list)):
    h_layer1 = np.int(parameter_list[i][0])
    h_layer2 = np.int(parameter_list[i][1])
    h_layer3 = np.int(parameter_list[i][2])
    path = cwd+'/pre_train_folder_'+str(h_layer1)+'_'+str(h_layer2)+'_'+str(h_layer3)+'dropout'  
    model = torch.load(path+'/'+str(h_layer1)+'_'+str(h_layer2)+'_'+str(h_layer3)+'_model_1')
    feature_size = model.sharedlayer[0].in_features  
    class PretrainedModel(nn.Module):
        def __init__(self):
            super(PretrainedModel, self).__init__()
            self.sharedlayer = nn.Sequential(
                nn.Linear(feature_size, h_layer1),
                nn.ReLU(),
                nn.Linear(h_layer1, h_layer2),
                nn.ReLU(),
                nn.Linear(h_layer2, h_layer3),
                nn.ReLU(),
                #nn.Linear(h_layer3, out_layer),
            )
            self.finallayer = nn.Sequential(nn.Linear(h_layer3,out_layer))
            self.sharedlayer[0].weight = torch.nn.Parameter(model.sharedlayer[0].weight)
            #self.sharedlayer[0].weight.requires_grad = False
            self.sharedlayer[0].bias = torch.nn.Parameter(model.sharedlayer[0].bias)
            #self.sharedlayer[0].bias.requires_grad = False
            self.sharedlayer[2].weight = torch.nn.Parameter(model.sharedlayer[2].weight)
            #self.sharedlayer[4].weight.requires_grad = False
            self.sharedlayer[2].bias = torch.nn.Parameter(model.sharedlayer[2].bias)
            #self.sharedlayer[4].bias.requires_grad = False
            self.sharedlayer[4].weight = torch.nn.Parameter(model.sharedlayer[4].weight)
            #self.sharedlayer[8].weight.requires_grad = False
            self.sharedlayer[4].bias = torch.nn.Parameter(model.sharedlayer[4].bias)
            #self.sharedlayer[8].bias.requires_grad = False
        def forward(self, x):
            x = self.sharedlayer(x)
            h_shared  = self.finallayer(x)
            return h_shared

    net_regr = NeuralNetRegressor(
        PretrainedModel,
        max_epochs=MAX_EPOCHS,
        optimizer = torch.optim.Adam,
        optimizer__weight_decay=0.001,
        batch_size = BATCH_SIZE,
        lr = LR,
        train_split = None,
        device='cuda',
    )
    X_regr = desc_target_scaler.astype(np.float32)
    y_regr = prop.astype(np.float32).values.reshape(-1, 1) 
    y_regr = np.log10(y_regr)
    
    f = open(path+'/performance'+'_'+tag,'w')
    Y_pred_list = []


    for j in range(15): # multiple training
        net_regr = NeuralNetRegressor(
            PretrainedModel,
            max_epochs=MAX_EPOCHS,
            optimizer = torch.optim.Adam,
            optimizer__weight_decay=0.0001,
            batch_size = BATCH_SIZE,
            lr = LR,
            train_split = None,
            device='cuda',
        )
        predicted = cross_val_predict(net_regr, X_regr, y_regr, cv=KFold(n_splits=5, shuffle=True))
        Y_pred_list.append(predicted)
        
        mae = mean_absolute_error(y_regr, predicted)
        mae_2 = mean_absolute_error(np.power(10,y_regr), np.power(10,predicted))		
        r2 = r2_score(y_regr, predicted)
        r2_2 = r2_score(np.power(10,y_regr), np.power(10,predicted))
        f.write('{:d} {:d} {:d} {:.8f} {:.8f} {:.8f} {:.8f}\n'.format(h_layer1, h_layer2, h_layer3, mae, r2, mae_2, r2_2))
    f.close()
    np.savetxt(path+'/Y_pred_'+tag, np.array(Y_pred_list).reshape(-1,df['thermal'].shape[0]), delimiter=',')
