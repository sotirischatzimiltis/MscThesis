import warnings
from collections import OrderedDict

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from tqdm import tqdm

import pandas as pd

# #############################################################################
# 1. Read and Prepare Data
# #############################################################################
train_Dataset = 'train_binary.csv'
test_Dataset = 'test_binary.csv'

class MyDataset(Dataset):

  def __init__(self,file_name):
    price_df=pd.read_csv(file_name)

    x=price_df.iloc[:,:-1].values
    y=price_df.iloc[:,-1].values
    print(x)
    print(y)
    self.x_train=torch.tensor(x,dtype=torch.float32)
    self.y_train=torch.tensor(y,dtype=torch.long)

  def __len__(self):
    return len(self.y_train)
  
  def __getitem__(self,idx):
    return self.x_train[idx],self.y_train[idx]


batch_size = 64
dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

trainDs=MyDataset(train_Dataset)
testDs=MyDataset(test_Dataset)
train_loader = DataLoader(trainDs, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(testDs, batch_size=batch_size, shuffle=True, drop_last=True)


# #############################################################################
# 2. Define netwrok as a regular PyTorch pipeline
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input_fc = nn.Linear(input_dim, 250)
        self.hidden_fc = nn.Linear(250, 100)
        self.output_fc = nn.Linear(100, output_dim)

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
    optimizer = optim.Adam(net.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    """Train the model on the training set."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)
    criterion = criterion.to(device)
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_acc = 0
        net.train()
        train_batch = iter(train_loader)
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
        print("Cross entropy loss: ",epoch_loss)
        print("Accurarcy:  ", epoch_acc/total)

def test(net, testloader):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
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


# #############################################################################
# 3. Federation of the pipeline with Flower
# #############################################################################

# Load model and data (simple CNN, CIFAR-10)
net = MLP(122,2).to(DEVICE)
#train(net,train_loader,10)
#loss, accuracy = test(net,test_loader)
#print("Loss: ", loss)
#print("Accuracy: ",accuracy)

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, train_loader, epochs=1)
        return self.get_parameters(), len(train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, test_loader)
        return loss, len(test_loader.dataset), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client("[::]:8080", client=FlowerClient())
