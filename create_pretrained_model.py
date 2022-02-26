import torch
import torch.nn as nn
from torch.autograd import Variable
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt
from tqdm import tqdm
import winsound


# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

baseline_perf = pd.DataFrame(index=np.arange(0, 1000))
baseline_training = pd.DataFrame(index=np.arange(0, 500))

# hyper parameters
input_size = 12  # 28x28
hidden_size = 20  # how many neurons per hidden layer
num_out = 4  # digits 0-9
num_epochs = 250

gamma = .98
learning_rate = 0.001

batch_size = 100

dfx = pd.read_csv('Data/X_pretrain.csv')
for column in dfx.columns:
    dfx[column] = (dfx[column] - dfx[column].min()) / (dfx[column].max() - dfx[column].min())
X = torch.nan_to_num(torch.tensor(dfx.iloc[1:4000, :].values, dtype=torch.float32))
dfy = pd.read_csv('Data/Y_pretrain.csv')[['Ca', 'Cb', 'Cc', 'Qout']]
for column in dfy.columns:
    dfy[column] = (dfy[column] - dfy[column].min()) / (dfy[column].max() - dfy[column].min())
Y = torch.nan_to_num(torch.tensor(dfy.iloc[1:4000, :].values, dtype=torch.float32))

X, Y = Variable(X).to(device), Variable(Y).to(device)

train_dataset = Data.TensorDataset(X, Y)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


class NeuralNet(nn.Module):
    def __init__(self, in_, hidden_, out_):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(in_, hidden_)  # Linear combinations for layer 1
        self.l1.cuda()
        self.relu1 = nn.LeakyReLU()  # Hidden layer activation function
        self.lin = nn.Linear(hidden_, hidden_)
        self.lin.cuda()
        self.relu2 = nn.LeakyReLU()  # Hidden layer activation function
        self.lin2 = nn.Linear(hidden_, hidden_)
        self.lin2.cuda()
        self.sig = nn.Tanh()
        self.sig.cuda()
        self.l2 = nn.Linear(hidden_, out_)  # Linear combinations for layer 2
        self.l2.cuda()

    def forward(self, x):
        out = self.l1(x)  # combination from inputs to hidden
        out = self.relu1(out)  # call the hidden layers activation function
        out = self.lin(out)  # call the hidden layers activation function
        out = self.relu2(out)  # call the hidden layers activation function
        out = self.lin2(out)
        out = self.sig(out)
        out = self.l2(out)  # combination from hidden to output
        # softmax applied automatically in main  by nn.CrossEntropyLoss()
        return out


# model
model = NeuralNet(input_size, hidden_size, num_out)

# set loss method
criterion = torch.nn.MSELoss()  # applies MSE and computes loss

# set optimizer method
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
# training loop
n_total_steps = len(train_loader)

pbar = tqdm(range(num_epochs), ncols=90)

for epoch in pbar:
    for step, (batch_x, batch_y) in enumerate(train_loader):  # for each training step

        b_x = Variable(batch_x)
        b_y = Variable(batch_y)

        prediction = model(b_x)     # input x and predict based on x

        loss = criterion(prediction, b_y)     # must be (1. nn output, 2. target)

        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients
    scheduler.step()
    pbar.set_postfix_str(f"loss: {loss.item():.6f}, lr: {scheduler.get_last_lr()[-1]:.3f}")


X2 = torch.nan_to_num(torch.tensor(dfx.iloc[4000:-1, :].values, dtype=torch.float32))
Y2 = torch.nan_to_num(torch.tensor(dfy.iloc[4000:-1, :].values, dtype=torch.float32))

X2, Y2 = Variable(X2).to(device), Variable(Y2).to(device)

pred = model(X2)

perf = (pred.data.cpu().numpy() - dfy.iloc[4000:-1, :].values)

print(np.mean((pred.data.cpu().numpy() - dfy.iloc[4000:-1, :].values)**2))

torch.save(model.state_dict(), 'pretrained_model.pt')
