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

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
input_size = 12  # 28x28
hidden_size = 20  # how many neurons per hidden layer
num_out = 2  # digits 0-9
num_epochs = 1000
batch_size = 100
gamma = .99
learning_rate = .01

mask = np.arange(100, 600, 5)
print(len(mask))

dfx = pd.read_csv('Data/Xtrain_10000.csv')
print(dfx.columns)
for column in dfx.columns:
    dfx[column] = (dfx[column] - dfx[column].min()) / (dfx[column].max() - dfx[column].min())
X = torch.tensor(dfx.iloc[mask, :].values, dtype=torch.float32)
dfy = pd.read_csv('Data/Ytrain_10000.csv')[['Ca', 'Cb', 'Cc', 'Qout']]
for column in dfy.columns:
    dfy[column] = (dfy[column] - dfy[column].min()) / (dfy[column].max() - dfy[column].min())
Y = torch.tensor(dfy.iloc[mask, :].values, dtype=torch.float32)


X, Y = Variable(X).to(device), Variable(Y).to(device)

train_dataset = Data.TensorDataset(X, Y)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_out):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)  # Linear combinations for layer 1
        self.l1.cuda()
        self.relu1 = nn.LeakyReLU()  # Hidden layer activation function
        self.lin = nn.Linear(hidden_size, hidden_size)
        self.lin.cuda()
        self.relu2 = nn.LeakyReLU()  # Hidden layer activation function
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin2.cuda()
        self.sig = nn.Tanh()
        self.sig.cuda()
        self.l2 = nn.Linear(hidden_size, num_out)  # Linear combinations for layer 2
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
    for step, (batch_x, batch_y) in enumerate(train_loader): # for each training step

        b_x = Variable(batch_x)
        b_y = Variable(batch_y)

        prediction = model(b_x)     # input x and predict based on x
        # print(prediction)

        Rrxn = prediction[:, 1]
        Qa, Qb = batch_x[:, 0], batch_x[:, 1]
        Ca0, Cb0 = batch_x[:, 4], batch_x[:, 5]
        A1, B1, C1 = batch_x[:, 6], batch_x[:, 7], batch_x[:, 8]
        V1 = batch_x[:, 10]
        Qout = prediction[:, 0]
        V2 = V1 + Qa + Qb - Qout
        A2 = torch.unsqueeze((Qa*Ca0 - Qout*A1 - Rrxn + (V2-V1)*A1 + V1*A1)/V2, dim=1)
        B2 = torch.unsqueeze((Qb*Cb0 - Qout*B1 - Rrxn + (V2-V1)*B1 + V1*B1)/V2, dim=1)
        C2 = torch.unsqueeze((0 - Qout*C1 + Rrxn + (V2-V1)*C1 + V1*C1)/V2, dim=1)
        Qout = torch.unsqueeze(Qout, dim=1)
        # print(A2.size(), B2.size(), C2.size(), Qout.size())
        pred = torch.cat((A2, B2, C2, Qout), dim=1)

        loss = criterion(pred, b_y)     # must be (1. nn output, 2. target)

        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients
    scheduler.step()
    pbar.set_postfix_str(f"loss: {loss.item():.6f}, lr: {scheduler.get_last_lr()[-1]:.6f}")

mask2 = np.arange(1000, 10000, 100)

dfx2 = pd.read_csv('Data/Xtrain_10000.csv')
print(dfx2.columns)
for column in dfx2.columns:
    dfx2[column] = (dfx2[column] - dfx2[column].min()) / (dfx2[column].max() - dfx2[column].min())
X2 = torch.tensor(dfx2.iloc[mask2, :].values, dtype=torch.float32)
dfy2 = pd.read_csv('Data/Ytrain_10000.csv')[['Ca', 'Cb', 'Cc', 'Qout']]
for column in dfy.columns:
    dfy2[column] = (dfy2[column] - dfy2[column].min()) / (dfy2[column].max() - dfy2[column].min())
Y2 = torch.tensor(dfy2.iloc[mask2, :].values, dtype=torch.float32)


X2, Y2 = Variable(X2).to(device), Variable(Y2).to(device)

pred = model(X2)

Rrxn = pred[:, 1]
Qa, Qb = X2[:, 0], X2[:, 1]
Ca0, Cb0 = X2[:, 4], X2[:, 5]
A1, B1, C1 = X2[:, 6], X2[:, 7], X2[:, 8]
V1 = X2[:, 10]
Qout = pred[:, 0]
V2 = V1 + Qa + Qb - Qout
A2 = torch.unsqueeze((Qa*Ca0 - Qout*A1 - Rrxn + V1*A1)/V2, dim=1)
B2 = torch.unsqueeze((Qb*Cb0 - Qout*B1 - Rrxn + V1*B1)/V2, dim=1)
C2 = torch.unsqueeze((0 - Qout*C1 + Rrxn + V1*C1)/V2, dim=1)
Qout = torch.unsqueeze(Qout, dim=1)
# print(A2.size(), B2.size(), C2.size(), Qout.size())
pred = torch.cat((A2, B2, C2, Qout), dim=1)

t = range(len(pred))

print(Y2.size())
print(pred.size())
print(t)
fig, ax = plt.subplots(2, 2)
fig.suptitle('Baseline')
for i in range(4):
    ax[int(i/2), i%2].set_title(dfy.columns[i])
    ax[int(i/2), i%2].plot(t, Y2.T[i].data.cpu().numpy()-pred.T[i].data.cpu().numpy())
plt.show()
