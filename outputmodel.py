import torch
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
class Net(torch.nn.Module):
        def __init__(self, n_feature, n_hidden,n_hidden2,n_hidden3,n_output):
            super(Net, self).__init__()
            self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
            self.hidden2 = torch.nn.Linear(n_hidden,n_hidden2)
            self.hidden3 = torch.nn.Linear(n_hidden2,n_hidden3)
            self.predict = torch.nn.Linear(n_hidden3, n_output)   # output layer

        def forward(self, x):
            x = F.sigmoid(self.hidden(x))      # activation function for hidden layer
            x = F.sigmoid(self.hidden2(x))
            x = F.sigmoid(self.hidden3(x))
            x = self.predict(x)             # linear output
            return x
net = Net(n_feature=27, n_hidden=18,n_hidden2=18,n_hidden3=18, n_output=9)     # define the network
try:
	net.load_state_dict(torch.load('net.pkl'))
	print(net)
	weights = list(net.hidden.parameters())[0].detach().numpy()
	bias = list(net.hidden.parameters())[1].detach().numpy()
	f = open('model_new.txt','w')
	for i in range(36,36+18*3):
		if(i == 36+18*3-1):
			f.write('{}\n'.format(i))
		else:
			f.write('{} '.format(i))
	#print(weights)
	for i in range(0,27):
		for j in range(36,36+18):
			f.write('{} {} {}\n'.format(i,j,str(weights[j-36][i])))
			f.write('{} {} {}\n'.format(j,j,str(bias[j-36])))
	weights = list(net.hidden2.parameters())[0].detach().numpy()
	bias = list(net.hidden2.parameters())[1].detach().numpy()
	for i in range(36,36+18):
		for j in range(36+18,36+18*2):
			f.write('{} {} {}\n'.format(i,j,str(weights[j-36-18][i-36])))
			f.write('{} {} {}\n'.format(j,j,str(bias[j-36-18])))
	weights = list(net.hidden3.parameters())[0].detach().numpy()
	bias = list(net.hidden3.parameters())[1].detach().numpy()
	for i in range(36+18,36+18*2):
		for j in range(36+18*2,36+18*3):
			f.write('{} {} {}\n'.format(i,j,str(weights[j-36-18*2][i-36-18])))
			f.write('{} {} {}\n'.format(i,j,str(bias[j-36-18*2])))
	weights = list(net.predict.parameters())[0].detach().numpy()
	bias = list(net.predict.parameters())[1].detach().numpy()
	for i in  range(36+18*2,36+18*3):
		for j in range(27,36):
			f.write('{} {} {}\n'.format(i,j,str(weights[j-27][i-36-18*2])))
			f.write('{} {} {}\n'.format(i,j,str(bias[j-27])))
#	print(list(net.hidden2.parameters())[1].detach().numpy())
except Exception as e:
	raise e




