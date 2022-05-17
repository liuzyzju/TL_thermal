import os
import numpy as np 
import pandas as pd
'''
cwd = os.getcwd()
f = open(cwd+'/FineTuned_Averaged_performance_0.1', 'w')

parameter_list = np.loadtxt('parameter_list')
for i in range(1,len(parameter_list)):
	h_layer1 = np.int(parameter_list[i][0])
	h_layer2 = np.int(parameter_list[i][1])
	h_layer3 = np.int(parameter_list[i][2])
	path = cwd+'/pre_train_folder_'+str(h_layer1)+'_'+str(h_layer2)+'_'+str(h_layer3)+'dropout'
	performance = np.loadtxt(path+'/'+'performance_transfer_run')
	performance_proxy = np.loadtxt(path+'/'+'performance_1')
	performance_nonTL = np.loadtxt(path+'/'+'performance_nontransfer_run')
	f.write('{:03d} {:03d} {:03d} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f}\n'.format(h_layer1, h_layer2, h_layer3, np.mean(performance_proxy[:,5]), np.mean(performance_proxy[:,6]), np.mean(performance[:,5]), np.std(performance[:,5]), np.mean(performance[:,6]), np.std(performance[:,6]),  np.mean(performance_nonTL[:,5]),  np.std(performance_nonTL[:,5]),np.mean(performance_nonTL[:,6]),  np.std(performance_nonTL[:,6])))

# model3_test_performance_test_2
# Proxy mae, proxy r2, target mae mean, target mae std, target r2 mean, target r2 std
'''
df = pd.read_csv('/afs/crc.nd.edu/user/z/zliu10/DeepLearning/agl_thermal/brand_new/Property_target.csv')
df = df.drop(columns = ['Unnamed: 0'])
prop = df['thermal'] # Target properties

cwd = os.getcwd()
f = open(cwd+'/afd_overall', 'w')

def Afd(a,b):
    temp = np.power(10,np.mean(np.abs(np.log10(a) - np.log10(np.mean(np.power(10,b),axis=0)))))
    return temp

parameter_list = np.loadtxt('parameter_list')
for i in range(1,len(parameter_list)):
	h_layer1 = np.int(parameter_list[i][0])
	h_layer2 = np.int(parameter_list[i][1])
	h_layer3 = np.int(parameter_list[i][2])
	path = cwd+'/pre_train_folder_'+str(h_layer1)+'_'+str(h_layer2)+'_'+str(h_layer3)+'dropout'
	y_non = np.loadtxt(path+'/Y_pred_nontransfer_run',delimiter=',')
	y_tl = np.loadtxt(path+'/Y_pred_transfer_run',delimiter=',')
	afd_non = Afd(prop,y_non)
	afd_tl = Afd(prop,y_tl)
	#performance = np.loadtxt(path+'/'+'performance_transfer_run')
	#performance_proxy = np.loadtxt(path+'/'+'performance_1')
	#performance_nonTL = np.loadtxt(path+'/'+'performance_nontransfer_run')
	f.write('{:03d} {:03d} {:03d} {:.8f} {:.8f}\n'.format(h_layer1, h_layer2, h_layer3, afd_non, afd_tl))
