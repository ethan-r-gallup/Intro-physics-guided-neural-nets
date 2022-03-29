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
import time
import plant_test_hybrid
import baseline
import transfer_learning
import Loss

torch.autograd.set_detect_anomaly(True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

skips = [1, 3, 6, 30, 60, 120]
batch = [99, 25, 11, 9, 6, 4]

for j in range(len(batch)):
    tran = np.array([])
    loss = np.array([])

    for i in tqdm(range(20)):
        loss = np.append(loss, Loss.run_loss(batch[j], skips[j]))
        tran = np.append(tran, transfer_learning.run_trans(batch[j], skips[j]))

    tran = pd.DataFrame(tran)
    tran.to_csv(f'perf_data2.transfer_{skips[j]}.csv')
    loss = pd.DataFrame(loss)
    loss.to_csv(f'perf_data2.loss_{skips[j]}.csv')


