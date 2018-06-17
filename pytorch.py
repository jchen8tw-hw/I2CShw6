"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
Dependencies:
torch: 0.4
matplotlib
"""
import torch
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable

torch.manual_seed(1)    # reproducible
BATCH_SIZE = 2170
#x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
#y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)

npx = np.loadtxt('problem/input_x_new.txt')
npy = np.loadtxt('problem/input_y_new.txt')
x = torch.from_numpy(npx)
y = torch.from_numpy(npy)
x.dtype = torch.float32
y.dtype = torch.float32
#x = torch.linspace(1, 10, 20)       # this is x data (torch tensor)
#y = torch.linspace(10, 1, 20) # this is y data (torch tensor)
print(x)
print(y)
torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # random shuffle for training
    num_workers=2,              # subprocesses for loading data
)
print(torch_dataset)
# torch can only train on Variable, so convert them to Variable
# The code below is deprecated in Pytorch 0.4. Now, autograd directly supports tensors
# x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()


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
print(net)  # net architecture

optimizer = torch.optim.Adam(net.parameters(), lr=0.01,betas=(0.9, 0.99))
loss_func = torch.nn.CrossEntropyLoss()
#loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

for epoch in range(3):   # train entire dataset 3 times
    for step, (batch_x, batch_y) in enumerate(loader):  # for each training step
        # train your data...
        batch_x = Variable(batch_x)
        batch_y = Variable(batch_y)
        prediction = net(batch_x)
        loss = loss_func(prediction,batch_y)
        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients
        print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
        batch_x.numpy(), '| batch y: ', batch_y.numpy())
        


#plt.ion()   # something about plotting
"""
for t in range(2000):
    prediction = net(x)     # input x and predict based on x

    loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients

    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)
"""
plt.ioff()
plt.show()