#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('reset', '-f')
import torch
from torchvision import transforms
from matplotlib import pyplot as plt
from models import TCNmodel
import numpy as np
from random import randrange
import os

train_sample,frame = torch.load('dataset_train.pt')

def DatasetCon(data,frame):
    # construct a F x 3 x C x W x H tensoR
    # 0: anchor, 1: positive, 2: negative
    N,F,C,W,H = data.shape # F: frame
    anchor = data
    idx = np.arange(F)
    count=0

    # positve & negative
    posindex = np.array([])
    negindex = np.array([])
    for j in range(len(frame)):
        f = frame[j]
        Posidx_j = np.zeros(f)
        Negidx_j = np.zeros(f)
        for i in range(f):
        # pp positve
            pp= randrange(-1,1)
            if pp ==-1:
                Posidx_j[i]=-1
            else:
                Posidx_j[i]=1
        
        # pn negative      
            pn = randrange(0,f)
            while pn<=i+2 & pn>=i-2:
                pn=randrange(0,f)
            Negidx_j[i]=pn+count
        
        count = count+f
         
        # fixed index value at 0 frame and the end frame
        Posidx_j[0]=1
        Posidx_j[-1]=-1
        posindex=np.concatenate((posindex,Posidx_j))
        negindex = np.concatenate((negindex,Negidx_j))
    
    posindex = idx+posindex
    positive = data[:,posindex,:,:,:]

    negative = data[:,negindex,:,:,:]
    dataset = torch.cat([anchor,positive,negative])
    dataset = dataset.transpose(0,1)
    return dataset

def train(train_loader, net, optimizer, criterion,device):
    """
    Trains network for one epoch in batches.

    Args:
        train_loader: Data loader for training set.
        net: Neural network model.
        optimizer: Optimizer (e.g. SGD).
        criterion: Loss function (e.g. cross-entropy loss).
    """
  
    avg_loss = 0
    #correct = 0
    total = 0

    # iterate through batches
    for i, data in enumerate(train_loader):
        N,P,C,W,H = data.shape
        # get the inputs; data is a list of [inputs, labels]
        inputs = data.reshape([-1,C,W,H])
        inputs= inputs.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        outputs = outputs.reshape([N,P,-1])
        
        anchor = outputs[:,0,:]
        positive = outputs[:,1,:]
        negative = outputs[:,2,:]
        
        loss = criterion(anchor,positive,negative)
        loss.backward()
        optimizer.step()

        # keep track of loss and accuracy
        avg_loss += loss
        #print(avg_loss)
    return avg_loss


os.makedirs("./checkpoints/", exist_ok=True)
from tqdm import tqdm
epochs = 1000

# Create instance of Network
net = TCNmodel()

# Create loss function and optimizer
criterion = torch.nn.TripletMarginLoss(margin=0.2, p=2,reduction='sum')
optimizer = torch.optim.SGD(net.parameters(), lr=1e-3,momentum=0.9)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = net.to(device)

# ini
check = 0
loss = 2

# load pretrained data
# check = 650 # enter the latest epoch number you had !! need to enter manually else start with -1
# PATH = './checkpoints/{}_{}.pth'.format('SaveModel',check)
# net.load_state_dict(torch.load(PATH)['net'])
# optimizer.load_state_dict(torch.load(PATH)['optimizer'])
# loss = torch.load(PATH)['minloss']
# print(loss)
# print(device)
for epoch in tqdm(range(epochs)):  # loop over the dataset multiple times
    # randomly generate pair dataset
    dataset = DatasetCon(train_sample,frame)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size = 4, shuffle = True)
    # Train on data
    train_loss = train(data_loader,net,optimizer,criterion,device)
    print('average loss:',train_loss/421,' min loss:', loss)
    if (loss>train_loss.item()/421) | (((epoch+1) % 20 == 0) and epoch):
        loss = min(train_loss.item()/421,loss)
        torch.save({'minloss':loss,
                    'loss':train_loss/421,
            'net': net.state_dict(),
            'optimizer':optimizer.state_dict()
            }, './checkpoints/{}_{}.pth'.format('SaveModel', epoch+1+check))


# In[ ]:


print((3>train_loss.item()))


# In[ ]:


train_loss.item()


# In[9]:


min(0,train_loss.item())


# In[ ]:




