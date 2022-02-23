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

phyloss_training = pd.DataFrame(index=np.arange(0, 500))

for skipsize in tqdm(range(1, 100, 1), ncols=100):
    # hyper parameters
    input_size = 12  # 28x28
    hidden_size = 20  # how many neurons per hidden layer
    num_out = 4  # digits 0-9
    num_epochs = 500
    gamma = .995
    learning_rate = .01

    mask = np.arange(100, 3100, skipsize)
    lm = len(mask)

    batch_size = int(8.87509976e-9*lm**3 + 2.74530619e-06*lm**2 + 1.39164881e-02*lm + 9.58001062)

    dfx = pd.read_csv('Data/Xtrain_10000.csv')
    for column in dfx.columns:
        dfx[column] = (dfx[column] - dfx[column].min()) / (dfx[column].max() - dfx[column].min())
    X = torch.tensor(dfx.iloc[mask, :].values, dtype=torch.float32)
    dfy = pd.read_csv('Data/Ytrain_10000.csv')[['Ca', 'Cb', 'Cc', 'Qout', 'V', 'dVdt']]
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

    # pbar = tqdm(range(num_epochs), ncols=90)

    losses = pd.DataFrame(columns=[f'{skipsize}'])

    for epoch in range(num_epochs):
        for step, (batch_x, batch_y) in enumerate(train_loader): # for each training step

            b_x = Variable(batch_x)
            b_y = Variable(batch_y[:, 0:4])
            # print('y', b_y.size())

            prediction = model(b_x)     # input x and predict based on x
            # print('pred', prediction.size())
            Qa, Qb, Qout = batch_x[:, 0], batch_x[:, 1], batch_y[:, 3]
            Ca0, Cb0 = batch_x[:, 4], batch_x[:, 5]
            Af, Bf, Cf = prediction[:, 0], prediction[:, 1], prediction[:, 2]
            Ai, Bi, Ci = batch_x[:, 6], batch_x[:, 7], batch_x[:, 8]
            V1, V2 = batch_x[:, 10], batch_y[:, 4]

            in_ = Qa*Ca0 + Qb*Cb0
            out_ = Qout*(Af+Bf+Cf)
            gen_ = V2*Cf - V1*Ci
            cons_ = (V2*Af - V1*Ai) + (V2*Bf - V1*Bi)
            acc_ = V2*(Af + Bf + Cf) - V1*(Ai + Bi + Ci)
            zero_ = torch.mean(((in_ - out_ + gen_ - cons_) - acc_)**2)
            m = torch.nn.ReLU()
            zero_ = (m(.00000000001-zero_)+m(-zero_-.00000000001))

            l = criterion(prediction, b_y)  # must be (1. nn output, 2. target)

            loss = l + 3*zero_

            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients
        scheduler.step()
        losses = losses.append({f'{skipsize}': l.data.cpu().numpy()}, ignore_index=True)
        # pbar.set_postfix_str(f"loss: {l.item():.6f}, lr: {scheduler.get_last_lr()[-1]:.3f}, phy: {zero_.item():.6f}")
    phyloss_training = pd.concat([phyloss_training, losses], axis=1)

    # baseline_training.join(losses)
    # plt.plot(losses)
    # plt.show()

    dfx2 = pd.read_csv('Data/Xtrain_10000.csv')
    for column in dfx2.columns:
        dfx2[column] = (dfx2[column] - dfx2[column].min()) / (dfx2[column].max() - dfx2[column].min())
    X2 = torch.tensor(dfx2.iloc[3100:-1, :].values, dtype=torch.float32)
    dfy2 = pd.read_csv('Data/Ytrain_10000.csv')[['Ca', 'Cb', 'Cc', 'Qout']]
    for column in dfy2.columns:
        dfy2[column] = (dfy2[column] - dfy2[column].min()) / (dfy2[column].max() - dfy2[column].min())
    Y2 = torch.tensor(dfy2.iloc[3100:-1, :].values, dtype=torch.float32)

    X2, Y2 = Variable(X2).to(device), Variable(Y2).to(device)

    pred = pd.DataFrame(model(X2).data.cpu().numpy(), columns=['Ca_pred', 'Cb_pred', 'Cc_pred', 'Qout_pred'])

    actual = dfy2.iloc[3100:-1, :].reset_index()

    phyloss_perf = pd.concat([pred, actual], axis=1)

    phyloss_perf.to_csv(f'performance data/phyloss_perf_{skipsize}4.csv')
    # print(skipsize, np.mean((phyloss_perf.pred - phyloss_perf.Cc)**2))
phyloss_training.to_csv('training data/phyloss_training4.csv')

# filename = 'tron.wav'
# winsound.PlaySound(filename, winsound.SND_FILENAME)
winsound.PlaySound('*', winsound.SND_ALIAS)
