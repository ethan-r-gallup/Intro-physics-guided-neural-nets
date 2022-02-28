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
num_epochs = 500
gamma = 0.99
learning_rate = 0.005

batch_size = 100

dfx = pd.read_csv('Data/Xtrain_10000.csv')
for column in dfx.columns:
    dfx[column] = (dfx[column] - dfx[column].min()) / (dfx[column].max() - dfx[column].min())
X = torch.tensor(dfx.iloc[0:3000, :].values, dtype=torch.float32)
dfy = pd.read_csv('Data/Ytrain_10000.csv')[['Ca', 'Cb', 'Cc', 'Qout']]
for column in dfy.columns:
    dfy[column] = (dfy[column] - dfy[column].min()) / (dfy[column].max() - dfy[column].min())
Y = torch.tensor(dfy.iloc[0:3000, :].values, dtype=torch.float32)

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
model.load_state_dict(torch.load('pretrained_model.pt'), strict=False)

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

        loss = criterion(prediction, b_y)     # must be (1. nn output, 2. target)

        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients
    scheduler.step()
    pbar.set_postfix_str(f"loss: {loss.item():.6f}, lr: {scheduler.get_last_lr()[-1]:.8f}")

# baseline_training.join(losses)
# plt.plot(losses)
# plt.show()

dfx2 = pd.read_csv('Data/Xtrain_10000.csv')
# print(dfx2.columns)
for column in dfx2.columns:
    dfx2[column] = (dfx2[column] - dfx2[column].min()) / (dfx2[column].max() - dfx2[column].min())
X2 = torch.tensor(dfx2.iloc[3100:-1, :].values, dtype=torch.float32)
dfy2 = pd.read_csv('Data/Ytrain_10000.csv')[['Ca', 'Cb', 'Cc', 'Qout']]
for column in dfy.columns:
    dfy2[column] = (dfy2[column] - dfy2[column].min()) / (dfy2[column].max() - dfy2[column].min())
Y2 = torch.tensor(dfy2.iloc[3100:-1, :].values, dtype=torch.float32)

X2, Y2 = Variable(X2).to(device), Variable(Y2).to(device)

pred = pd.DataFrame(model(X2).data.cpu().numpy(), columns=['Ca_pred', 'Cb_pred', 'Cc_pred', 'Qout_pred'])

