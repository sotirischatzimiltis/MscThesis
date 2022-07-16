#*****************************************************************************#
# Imports
from typing import List, Tuple
from collections import OrderedDict
import warnings
import pandas as pd

import flwr as fl
from flwr.common import Metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import confusion_matrix 
#*****************************************************************************#
# 1. Data declaration
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
#*****************************************************************************#
# 2. Configure FFNNetwork 
class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()
        self.input_fc = nn.Linear(input_dim, 200)
        self.hidden_fc = nn.Linear(200, 200)
        self.output_fc = nn.Linear(200, output_dim)
        
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        h_1 = F.relu(self.input_fc(x))
        h_2 = F.relu(self.hidden_fc(h_1))
        y_pred = self.output_fc(h_2)
        return y_pred, h_2
#*****************************************************************************#
# 3. Server-side / Centralised Evaluation
def get_eval_fn(model):
    test_Dataset = "C:/Users/sotir/Desktop/Masters/Dissertation/DATA/Binary_Classification/StandardScaler/All_features/test_binary.csv"
    batch_size = 1024
    testDs=MyDataset(test_Dataset)
    testloader = DataLoader(testDs, batch_size=batch_size, shuffle=True, drop_last=False)

    # The `evaluate` function will be called after every round
    def evaluate(weights: fl.common.Weights):
        params_dict = zip(model.state_dict().keys(), weights)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict,strict = True) # set weights to model 
        loss, accuracy, tn, fp, fn, tp = evaluate_centralised(model,testloader) # evaluate the model 
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
    
def evaluate_centralised(model, testloader):
    loss_fn = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    y_p = [] 
    y_test = []
    with torch.no_grad():
        for data, targets in tqdm(testloader):
            data = data.to(device)
            targets = targets.to(device)
            y_test.append(targets)
            y_pred,_ = model(data)
            y_p.append(torch.max(y_pred.data, 1)[1]) # concatenate tensors 
            loss += loss_fn(y_pred,targets).item()
            total += targets.size(0)
            correct += (torch.max(y_pred.data, 1)[1] == targets).sum().item()
    y_p = torch.cat(y_p) # concatenate the multiple tensors into 1 tensor 
    y_test = torch.cat(y_test) # concatenate the multiple tensors into 1 tensor 
    y_p = y_p.cpu().detach().numpy() # convert tensor to numpy array
    y_test = y_test.cpu().detach().numpy()
    tn, fp, fn, tp = confusion_matrix(y_test, y_p).ravel()       
    return loss / len(testloader.dataset), correct / total, tn, fp, fn, tp
#*****************************************************************************#
# 4. Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy from weighted_average": sum(accuracies) / sum(examples)}
#*****************************************************************************#
# Main
warnings.filterwarnings("ignore", category=UserWarning)
# if gpu is available then use it 
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 1024
num_of_features = 122
num_of_outputs = 2
# create model to be use for centralised evaluation 
model = Net(num_of_features,num_of_outputs).to(DEVICE)
# Define strategy
strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average,eval_fn = get_eval_fn(model)
,fraction_fit=1,min_fit_clients=3,min_available_clients=3,min_eval_clients=3)
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