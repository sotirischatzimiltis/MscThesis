#*****************************************************************************#
# Imports
import warnings
from collections import OrderedDict
from typing import List, Tuple
import flwr as fl
from flwr.common import Metrics

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
from sklearn.metrics import confusion_matrix 
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
# 2. Data Preparation  
class MyDataset(Dataset):
  def __init__(self,file_name):
    df=pd.read_csv(file_name)
    x=df.iloc[:,:-1].values
    y=df.iloc[:,-1].values
    self.x_test=torch.tensor(x,dtype=torch.float32)
    self.y_test=torch.tensor(y,dtype=torch.long)
    
  def __len__(self):
    return len(self.y_test)
  
  def __getitem__(self,idx):
    return self.x_test[idx],self.y_test[idx]
    
test_Dataset = "C:/Users/sotir/Desktop/Masters/Dissertation/DATA/Binary_Classification/StandardScaler/All_features/test_binary.csv"
testDs=MyDataset(test_Dataset)
batch_size = 256
testloader = DataLoader(testDs, batch_size=batch_size, shuffle=True, drop_last=True)
#*****************************************************************************#        
# 3. Server Side Evaluation
def get_eval_fn(net,testloader,loss_fn,num_steps):
    """Return an evaluation function for server-side evaluation."""

    # The `evaluate` function will be called after every round
    def evaluate(weights: fl.common.Weights):
        params_dict = zip(net.state_dict().keys(), weights)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict,strict = True) # set weights to net 
        loss, accuracy, tn, fp, fn, tp = evaluate_centralised(net,testloader,loss_fn,num_steps) # evaluate the net 
        precision = 100 * (tp / (tp + fp))
        recall = 100 * (tp / (tp + fn)) 
        far = 100 * (fp / (fp + tn)) # false alarm rate %
        dr = 100 * (tp / (tp + fn)) # detection rate % 
        f1_score = 100 * ((2 * precision * recall) / (precision + recall)) # f1 score %
        fpr = 100 * (fp / (fp + tp)) # false positive rate
        fnr = 100 * (fn / (fn + tn)) # false negative rate
        return loss, {"accuracy": accuracy, "precision":precision,"recall":recall,"f1_score":f1_score,"detection_rate":dr,
        "false_alarm_rate":far,"false_positive_rate":fpr,"false_negative_rate":fnr,"true_negatives":tn,"false_positives":fp,
        "false_negatives":fn,"true_positives":tp}

    return evaluate
    
def evaluate_centralised(net, testloader,loss_fn,num_steps):
    loss = 0.0
    test_acc = 0.0
    total = 0
    correct = 0
    y_pred = [] 
    y_test = [] 
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
            y_pred.append(predicted) # concatenate tensors 
            y_test.append(targets)
            correct += (predicted == targets).sum().item()

            # Test set loss
            test_loss = torch.zeros((1),device=DEVICE)
            for step in range(num_steps):
                test_loss += loss_fn(test_mem[step], targets)
            loss += test_loss.item()  
    #print(y_test)
    #print(y_pred)
    y_pred = torch.cat(y_pred) # concatenate the multiple tensors into 1 tensor 
    y_test = torch.cat(y_test) # concatenate the multiple tensors into 1 tensor 
    y_pred = y_pred.cpu().detach().numpy() # convert tensor to numpy array
    y_test = y_test.cpu().detach().numpy()
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    #print(y_pred)
    return loss / len(testloader.dataset), correct / total, tn, fp, fn, tp
    
# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy from weighted_average": sum(accuracies) / sum(examples)}
#*****************************************************************************#        
# 4. Declare/Initialize Variables - Main 
warnings.filterwarnings("ignore", category=UserWarning)
# if gpu is available then use it 
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# net used for centralised evaluation 
num_inputs = len(testDs.x_test[1]) # input layer neurons 
num_hidden = 100 # hidden layer neurons 
num_outputs = 2 # output layer neurons 
num_steps = 25 # temporal dynamics
beta = 0.9 # decay 
loss_fn = nn.CrossEntropyLoss() # loss definition
net = Net(num_inputs,num_hidden,beta).to(DEVICE)
# Define strategy
strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average,eval_fn = get_eval_fn(net,testloader,loss_fn,num_steps),
fraction_fit=1,min_fit_clients=3,min_available_clients=3,min_eval_clients=3)
# Start Flower server
fl.server.start_server(
    server_address="localhost:8080",
    config={"num_rounds": 5},
    strategy=strategy,
    certificates=(
        Path(".cache/certificates/ca.crt").read_bytes(),
        Path(".cache/certificates/server.pem").read_bytes(),
        Path(".cache/certificates/server.key").read_bytes(),
    )
)