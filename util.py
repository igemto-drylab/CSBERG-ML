import json
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split, ShuffleSplit
from scipy import stats
plt.rcParams['figure.dpi'] = 300


AA = ['a', 'r', 'n', 'd', 'c', 'q', 'e', 'g', 'h', 'i', 
      'l', 'k', 'm', 'f', 'p', 's', 't', 'w', 'y', 'v', '-']
AA_IDX = {AA[i]:i for i in range(len(AA))}
IDX_AA = {i:AA[i].upper() for i in range(len(AA))}


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def one_hot_encode_aa(aa_str, pad=None):
    aa_str = aa_str.lower()
    M = len(aa_str)
    aa_arr = np.zeros((M, 21), dtype=int)
    for i in range(M):
        aa_arr[i, AA_IDX[aa_str[i]]] = 1
    return aa_arr


def get_X(seqs):
    M = len(seqs[0])
    N = len(seqs)
    X = []
    for i in range(N):
        try:
            X.append(one_hot_encode_aa(seqs[i]))
        except KeyError:
            pass
    return np.array(X)


class SequenceData(Dataset):

    def __init__(self, X):
        if not torch.is_tensor(X):
            self.X = torch.from_numpy(X)
        else:
            self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


class SeqfuncData(Dataset):

    def __init__(self, X, y, scale_X=True):
        if not torch.is_tensor(X):
            self.X = torch.from_numpy(X)
        else:
            self.X = X
        if not torch.is_tensor(y):
            self.y = torch.from_numpy(y)
        else:
            self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def read_fasta(fname):
    seqs = []
    s = ""
    with open(fname) as f:
        line = f.readline()
        while line:
            if line.startswith(">"):
                if s != "":
                    seqs.append(s)
                s = ""
            elif len(line) > 0:
                s += line.strip()
            line = f.readline()
        seqs.append(s)

    X = torch.tensor(get_X(seqs))

    return X