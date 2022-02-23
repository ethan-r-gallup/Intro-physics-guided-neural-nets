import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from tqdm import tqdm
import pandas as pd
import numpy as np
import winsound

torch.autograd.set_detect_anomaly(True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

taylor_training = pd.DataFrame(index=np.arange(0, 500))

for skipsize in range(1, 11):
    # hyper parameters
    num_terms = 3
    input_size = 100
    num_epochs = 500
    learning_rate = .1
    gamma = .99
    span = 2
    center = 1

    mask = np.arange(100, 500, skipsize)
    batch_size = int(len(mask)/3)

    dfx = pd.read_csv('Data/Xtrain_10000.csv')
    for column in dfx.columns:
        dfx[column] = (dfx[column] - dfx[column].min()) / (dfx[column].max() - dfx[column].min())
    X = torch.tensor(dfx.iloc[mask, :].values, dtype=torch.float32)
    dfy = pd.read_csv('Data/Ytrain_10000.csv')[['Cc']]
    for column in dfy.columns:
        dfy[column] = (dfy[column] - dfy[column].min()) / (dfy[column].max() - dfy[column].min())
    Y = torch.tensor(dfy.iloc[mask, :].values, dtype=torch.float32)

    X, Y = Variable(X).to(device), Variable(Y).to(device)

    train_dataset = Data.TensorDataset(X, Y)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


    class Rrxn(nn.Module):
        """ Reaction Rate Series Layer """
        def __init__(self, n):
            super().__init__()
            bias = torch.rand(1)
            self.bias = nn.Parameter(bias)
            D = torch.rand([n])
            self.D = nn.Parameter(D)
            H = torch.rand([n, n])
            self.H = nn.Parameter(H)
            G = torch.rand([n, n, n])
            self.G = nn.Parameter(G)
            F = torch.rand([n, n, n, n])
            self.F = nn.Parameter(F)

        def forward(self, x):
            order1 = self.D @ torch.transpose(x, 0, 1)
            order2 = (1/2) * torch.einsum('ij,bi,jb->b', self.H, x, torch.transpose(x, 0, 1))
            x2 = torch.unsqueeze(x, 2)
            order3 = (1/6) * torch.einsum('ijk,bil,jbl,blk->b', self.G, x2, torch.transpose(x2, 0, 1), torch.transpose(x2, 1, 2))
            x3 = torch.unsqueeze(x2, 3)
            order4 = (1/24) * torch.einsum('ijkl,bimm,jbmm,bmkm,bmml->b', self.F, x3, torch.transpose(x3, 0, 1)
                                  , torch.transpose(x3, 1, 2), torch.transpose(x3, 1, 3))
            wx = self.bias + order1 + order2 + order3 + order4
            wx = torch.unsqueeze(wx, dim=1)
            return wx  # w times x + b


    class NeuralNet(nn.Module):
        def __init__(self, dim):
            super(NeuralNet, self).__init__()
            self.rate = Rrxn(dim)
            self.rate.cuda()

        def forward(self, x):
            out = self.rate(x)
            return out


    model = NeuralNet(12)

    # set loss method
    criterion = torch.nn.MSELoss()  # applies MSE and computes loss

    # set optimizer method
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[6,8,9], gamma=0.995)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[250, 500, 750, 1000, 1250, 1500, 1750], gamma=0.9)  # training loop
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.3)
    n_total_steps = len(train_loader)

    pbar = tqdm(range(num_epochs), ncols=140)

    losses = pd.DataFrame(columns=[f'{len(mask)}'])
    for epoch in pbar:
        for step, (batch_x, batch_y) in enumerate(train_loader): # for each training step

            b_x = Variable(batch_x, requires_grad=True)
            b_y = Variable(batch_y)

            prediction = model(b_x)     # input x and predict based on x
            loss = criterion(prediction, b_y)     # must be (1. nn output, 2. target)
            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients
        scheduler.step()
        losses = losses.append({f'{len(mask)}': loss.data.cpu().numpy()}, ignore_index=True)
        pbar.set_postfix_str(f"loss: {loss.item():.6f}, lr: {scheduler.get_last_lr()[-1]:.3f}")
    taylor_training = pd.concat([taylor_training, losses], axis=1)

    # taylor_training.join(losses)
    # plt.plot(losses)
    # plt.show()

    dfx2 = pd.read_csv('Data/Xtrain_10000.csv')
    print(dfx2.columns)
    for column in dfx2.columns:
        dfx2[column] = (dfx2[column] - dfx2[column].min()) / (dfx2[column].max() - dfx2[column].min())
    X2 = torch.tensor(dfx2.values, dtype=torch.float32)
    dfy2 = pd.read_csv('Data/Ytrain_10000.csv')[['Cc']]
    for column in dfy.columns:
        dfy2[column] = (dfy2[column] - dfy2[column].min()) / (dfy2[column].max() - dfy2[column].min())
    Y2 = torch.tensor(dfy2.values, dtype=torch.float32)

    X2, Y2 = Variable(X2).to(device), Variable(Y2).to(device)

    pred = pd.DataFrame(model(X2).data.cpu().numpy(), columns=['pred'])

    taylor_perf = pd.concat([pred, dfy2], axis=1)
    print('perf')
    print(taylor_perf)
    print(taylor_training)
    taylor_perf.to_csv(f'performance data/taylor_perf_{len(mask)}.csv')
taylor_training.to_csv('training data/taylor_training.csv')
