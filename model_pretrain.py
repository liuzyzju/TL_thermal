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


df = pd.read_csv('proxy_agl.csv')
df = df.drop(columns = ['Unnamed: 0'])
desc_hand = pd.read_csv(
	'proxy_desc_agl_mp.csv')
desc_hand = desc_hand.drop(columns = ['Unnamed: 0'])
prop = df['kL']

tag = '1' 


scaler = preprocessing.MinMaxScaler()
desc_el_scaler = scaler.fit_transform(desc_hand)

parameter_list = np.loadtxt('parameter_list')

device = 'cuda'

X = torch.from_numpy(np.array(desc_el_scaler)).type(torch.cuda.FloatTensor)
Y = torch.from_numpy(np.array(prop)).type(torch.cuda.FloatTensor)
Y = Y.reshape([len(Y),1])
Y = torch.log10(Y)

X = X.to(device)
Y = Y.to(device)

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

feature_size = desc_hand.shape[1]
out_layer = 1
LR = 0.001
BATCH_SIZE = int(len(df)/2)

for i in range(0,len(parameter_list)):
	h_layer1 = np.int(parameter_list[i][0])
	h_layer2 = np.int(parameter_list[i][1])
	h_layer3 = np.int(parameter_list[i][2])
	cwd = os.getcwd()
	path = cwd+'/pre_train_folder_'+str(h_layer1)+'_'+str(h_layer2)+'_'+str(h_layer3)+'dropout'
	if not os.path.exists(path):
		os.mkdir(path)

	torch_data  = torch.utils.data.TensorDataset(X, Y)
	loader = DataLoader(dataset=torch_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
	EPOCH = 1500
	test_net = TransNet()
	test_net = test_net.to(device)
	opt_adam = torch.optim.Adam(test_net.parameters(), lr=LR, weight_decay=0.001)
	loss_func = nn.MSELoss()
	loss_list = []
	for epoch in range(EPOCH):
		for step, (b_x, b_y) in enumerate(loader):
			b_x = b_x.float()
			b_y = b_y.float()
			pre = test_net(b_x)
			loss = loss_func(pre, b_y)
			opt_adam.zero_grad()
			loss.backward()
			opt_adam.step()
			loss_list.append(loss)
	# finished training
	loss_list = np.array(loss_list)
	torch.save(test_net,path+'/'+str(h_layer1)+'_'+str(h_layer2)+'_'+str(h_layer3)+'_model'+'_'+tag)
	#np.savetxt()
	np.savetxt(path+'/'+str(h_layer1)+'_'+str(h_layer2)+'_'+str(h_layer3)+'_lost'+'_'+tag, loss_list)

	# Evaluation of this parameter
	net_regr = NeuralNetRegressor(
    	TransNet,
    	max_epochs = EPOCH,
    	optimizer = torch.optim.Adam,
    	optimizer__weight_decay=0.001,
    	batch_size = BATCH_SIZE,
    	lr = LR,
    	train_split = None,
    	device='cuda',
	)

	f = open(path+'/'+'performance_'+tag, 'w')

	cycles = 5
	Y_pred_list = []

	for j in range(cycles):
		
		predicted = cross_val_predict(net_regr, X, Y, 
			cv=KFold(n_splits=5, shuffle=True))
		mae_2 = mean_absolute_error(np.power(10,Y.cpu()), np.power(10,predicted))
		mae = mean_absolute_error(Y.cpu(), predicted)		
		r2 = r2_score(Y.cpu(), predicted)
		r2_2 = r2_score(np.power(10,Y.cpu()), np.power(10,predicted))
		#Y_pred_sum += predicted
		f.write('{:d} {:d} {:d} {:.8f} {:.8f} {:.8f} {:.8f}\n'.format(h_layer1, h_layer2, h_layer3, mae, r2, mae_2, r2_2))
		Y_pred_list.append(predicted)
	f.close()
	np.savetxt(path+'/Y_pred_agl_list', np.array(Y_pred_list).reshape(-1,df['kL'].shape[0]), delimiter=',')
