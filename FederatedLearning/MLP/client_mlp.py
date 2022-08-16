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
import pandas as pd

from pathlib import Path
# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
# if gpu is available then use it 
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 64
# define the network --> this example uses a FFNN
class Net(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()
        self.input_fc = nn.Linear(input_dim, 200)
        self.hidden_fc = nn.Linear(200, 200)
        self.output_fc = nn.Linear(200, output_dim)
        
    def forward(self, x):
        # x = [batch size, height, width]
        #batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        # x = [batch size, height * width]
        h_1 = F.relu(self.input_fc(x))
        # h_1 = [batch size, 250]
        h_2 = F.relu(self.hidden_fc(h_1))
        # h_2 = [batch size, 100]
        y_pred = self.output_fc(h_2)
        # y_pred = [batch size, output dim]
        return y_pred, h_2

def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc        
            
def train(net, trainloader, epochs):
    optimizer = torch.optim.Adam(net.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    """Train the model on the training set."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)
    criterion = criterion.to(device)
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_acc = 0
        net.train()
        train_batch = iter(trainloader)
        total = 0 
        for data, targets in tqdm(train_batch):
           total += 1
           data = data.to(device)
           targets = targets.to(device)
           optimizer.zero_grad()
           y_pred,_ = net(data)
           loss = criterion(y_pred,targets)
           acc = calculate_accuracy(y_pred,targets)
           loss.backward()
           optimizer.step()
           epoch_loss += loss.item()
           epoch_acc += acc.item()
        #print("Cross entropy loss: ",epoch_loss)
        #print("Accurarcy:  ", epoch_acc/total)
    
def test(net, testloader):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        for data, targets in tqdm(testloader):
            data = data.to(device)
            targets = targets.to(device)
            y_pred,_ = net(data)
            loss += criterion(y_pred,targets).item()
            total += targets.size(0)
            correct += (torch.max(y_pred.data, 1)[1] == targets).sum().item()
    print("Loss: ",loss)
    print("Accuracy: ",correct/total)
    return loss / len(testloader.dataset), correct / total
    

class MyDataset(Dataset):

  def __init__(self,file_name):
    print("Collecting Data")
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

    trainDs=MyDataset(train_Dataset)
    testDs=MyDataset(test_Dataset)
    train_loader = DataLoader(trainDs, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(testDs, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_loader,test_loader
    


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Load model and data (simple FFNN, NSLKDD)
net = Net(122,2).to(DEVICE)
train_Dataset = "C:/Users/sotir/Desktop/Masters/Dissertation/DATA/FederatedLearning/Folds/3Fold/fold1train_data.csv" 
test_Dataset = "C:/Users/sotir/Desktop/Masters/Dissertation/DATA/FederatedLearning/Folds/3Fold/fold1test_data.csv"
trainloader, testloader = load_data(train_Dataset,test_Dataset,batch_size=64)

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
        train(net, trainloader, epochs=1)
        return self.get_parameters(), len(trainloader.dataset), {}
    # test the local model
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}


# Start Flower client
#fl.client.start_numpy_client("localhost:8080", client=FlowerClient(), root_certificates=Path(".cache/certificates/ca.crt").read_bytes(),)
fl.client.start_numpy_client("localhost:8080", client=FlowerClient())
