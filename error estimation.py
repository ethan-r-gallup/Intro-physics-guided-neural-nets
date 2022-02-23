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

d = 4  # m, reactor diameter
Cvalve = 3.2  # m^2.5/min, reactor valve constant
ko = .23  # m^3/kmol*min, rate constant
Ea = 9500  # kJ/kmol, reaction activation energy
R = 8.314  # kJ/kmol*K, ideas gas constant
rho = 780  # kh/m^3, density of all liquids
Cp = 3.25  # kJ/kg*K, Heat capacity of all liquids
dH = 156000  # kJ/kmol, heat of formation (per mole of A consumed)
q = 10


def reactor_model(Qa, Qb, Ta, Tb, Ca0, Cb0, Ca, Cb, Cc, T, V):
    global d, Cvalve, ko, Ea, R, q, rho, Cp
    h = V/((np.pi*d^2)/4) # m, height of fluid in tank

    Qout = Cvalve*np.sqrt(h) # m^3, quantity out

    Rrxn = ko*np.exp(-Ea/(R*T))*Ca*Cb*V # kmol/min, reaction rate

    dVdt = Qa+Qb-Qout # m^3/s change in volume
    dCa_dt = (Qa*Ca0-Rrxn-Qout*Ca-Ca*dVdt)/V  # (kmol/m^3)/s, change in [A]
    Ca = dCa_dt + Ca
    dCb_dt = (Qb*Cb0-Rrxn-Qout*Cb-Cb*dVdt)/V  # (kmol/m^3)/s, change in [B]
    Cb = dCb_dt + Cb
    dCc_dt = (Rrxn-Qout*Cc-Cc*dVdt)/V  # (kmol/m^3)/s, change in [C]
    Cc = dCc_dt + Cc

    q = q*1e6  # kJ, incoming heat

    dTdt = ((Qa*rho*Cp*Ta+Qb*rho*Cp*Tb-Qout*rho*Cp*T-Rrxn*dH+q)/(rho*Cp)-T*dVdt)/V  # K/s change in temperature

    return Ca, Cb, Cc, Qout


class FixError(nn.Module):
    def __init__(self, input_size, hidden_size, compressed_size, output_size):
        super(FixError, self).__init__()
        self.size_in = input_size
        self.conv1 = nn.Conv1d(input_size, 6, 2)
        self.conv1.cuda()
        self.conv2 = nn.Conv1d(6, 3, 2)
        self.conv2.cuda()
        self.pool = nn.MaxPool1d(2)
        self.pool.cuda()

        self.l1 = nn.Linear(output_size + compressed_size, hidden_size)  # Linear combinations for layer 1
        self.l1.cuda()
        self.relu = nn.LeakyReLU()  # Hidden layer activation function
        self.relu.cuda()
        self.lin = nn.Linear(hidden_size, hidden_size)
        self.lin.cuda()
        self.sig = nn.Tanh()
        self.sig.cuda()
        self.l2 = nn.Linear(hidden_size, output_size)  # Linear combinations for layer 2
        self.l2.cuda()

    def forward(self, x):
        compressed = F.relu(self.conv1(x[:, self.size_in+1]))
        compressed = self.pool(compressed)
        compressed = F.relu(self.conv2(compressed))
        compressed = self.pool(compressed)
        in_ = torch.cat((compressed, x[:, self.size_in:-1]))
        out_ = self.l1(in_)
        out_ = self.relu(out_)
        out_ = self.lin(out_)
        out_ = self.sig(out_)
        out = self.l2(out_)
        return out


mask = np.arange(100, 500, 5)
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


