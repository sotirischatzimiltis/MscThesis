#*****************************************************************************#
# Imports
import warnings
from collections import OrderedDict

import flwr as fl

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from tqdm import tqdm

import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import matplotlib.pyplot as plt
import numpy as np
import itertools
import pandas as pd 
#*****************************************************************************#
# 1. Define the SNN 
class Net(nn.Module):
    def __init__(self,num_inputs,num_hidden,beta):
        super().__init__()
        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.lif2 = snn.Leaky(beta=beta) 
        self.fc3 = nn.Linear(num_hidden,num_outputs)
        self.lif3 = snn.Leaky(beta=beta)

    def forward(self, x):
        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        # record final layer
        spk3_rec = []
        mem3_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3,mem3)
            spk3_rec.append(spk3)
            mem3_rec.append(mem3)

        return torch.stack(spk3_rec, dim=0), torch.stack(mem3_rec, dim=0)

#*****************************************************************************#
# 2. Training & Testing 
def print_batch_accuracy(data, targets, train=False):
    output, _ = net(data.view(batch_size, -1))
    _, idx = output.sum(dim=0).max(1)
    acc = np.mean((targets == idx).detach().cpu().numpy())
    if train:
        print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")
        
def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc        

def train(net,trainloader,epochs,loss_fn,optimizer):
    for epoch in range(epochs):
        train_loss = 0.0
        train_acc =0.0
        iter_counter = 0
        train_batch = iter(trainloader)
        train_samples = 0
        # Minibatch training loop
        for data, targets in tqdm(train_batch):
            train_samples += 1
            data = data.to(DEVICE)
            targets = targets.to(DEVICE)

            # forward pass
            net.train()
            spk_rec, mem_rec = net(data.view(batch_size, -1))
            _, idx = spk_rec.sum(dim=0).max(1)
            acc = np.mean((targets == idx).detach().cpu().numpy())
            train_acc+= acc
            # initialize the loss & sum over time
            loss_val = torch.zeros((1),device=DEVICE)
            for step in range(num_steps):
                loss_val += loss_fn(mem_rec[step], targets)

            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Store loss history for future plotting
            train_loss += loss_val.item()
            
def test(net, testloader,loss_fn):
    loss = 0.0
    test_acc = 0.0
    total = 0
    correct = 0
    with torch.no_grad(): # switch off gradient computation
        net.eval() # switch for evaluation 
        for data, targets in tqdm(testloader): # load batch of data and targets
            data = data.to(DEVICE) # sent to gpu
            targets = targets.to(DEVICE) # sent to gpu 
    
            # forward pass
            test_spk, test_mem = net(data.view(data.size(0), -1)) # forward pass to predict

            # calculate total accuracy
            _, predicted = test_spk.sum(dim=0).max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            # Test set loss
            test_loss = torch.zeros((1),device=DEVICE)
            for step in range(num_steps):
                test_loss += loss_fn(test_mem[step], targets)
            loss += test_loss.item()
            
    return loss / len(testloader.dataset), correct / total


#*****************************************************************************#        
# 3. Data Preparation  
class MyDataset(Dataset):
  def __init__(self,file_name):
    df=pd.read_csv(file_name)

    x=df.iloc[:,:-1].values
    y=df.iloc[:,-1].values
    self.x_train=torch.tensor(x,dtype=torch.float32)
    self.y_train=torch.tensor(y,dtype=torch.long)

  def __len__(self):
    return len(self.y_train)
  
  def __getitem__(self,idx):
    return self.x_train[idx],self.y_train[idx]
    
def load_data(train_Dataset,test_Dataset,batch_size):
    print("Collecting Data")
    trainDs=MyDataset(train_Dataset)
    testDs=MyDataset(test_Dataset)
    input_dim = len(trainDs.x_train[1])
    train_loader = DataLoader(trainDs, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(testDs, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_loader,test_loader,input_dim
#*****************************************************************************#        
# 4. Federated Learning using Flower

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    # return the model weight as a list of NumPy ndarrays
    def get_parameters(self):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]
    # update/set the local model weights with the parameters received from the server
    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
    # set the local model weights, train the local model,receive the updated local model weights
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, trainloader,epochs,loss_fn,optimizer)
        return self.get_parameters(), len(trainloader.dataset), {}
    # test the local model
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader,loss_fn)
        print("Cross Validation Accuracy: ",accuracy)
        print("Cross Validation Loss: ",loss)
        return loss, len(testloader.dataset), {"accuracy": accuracy}
#*****************************************************************************#
# Initialize variables
warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_Dataset = "C:/Users/sotir/Desktop/Masters/Dissertation/DATA/FederatedLearning/Folds/3Fold/fold1train_data.csv" 
test_Dataset = "C:/Users/sotir/Desktop/Masters/Dissertation/DATA/FederatedLearning/Folds/3Fold/fold1test_data.csv"
batch_size = 256 
trainloader, testloader, input_dim = load_data(train_Dataset,test_Dataset,batch_size=batch_size) # load data 
num_inputs = input_dim # input layer neurons 
num_hidden = 100 # hidden layer neurons 
num_outputs = 2 # output layer neurons 
num_steps = 25 # temporal dynamics
beta = 0.9 # decay 
learning_rate = 0.0005
net = Net(num_inputs,num_hidden,beta).to(DEVICE) # create network
loss_fn = nn.CrossEntropyLoss() # loss definition
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999)) # optimizer
epochs = 1
#*****************************************************************************#
# Start Flower client
#fl.client.start_numpy_client("localhost:8080", client=FlowerClient(), root_certificates=Path(".cache/certificates/ca.crt").read_bytes(),)
fl.client.start_numpy_client("localhost:8080", client=FlowerClient())
#*****************************************************************************#
