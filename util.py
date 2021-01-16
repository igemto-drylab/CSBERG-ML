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


def save_fasta(X_p, fname, sampling='max'):
    seqs = ""
    if torch.is_tensor(X_p):
        X_p = X_p.cpu().numpy()
    b, l, d = X_p.shape

    # nchar = 1
    for i in range(b):
        seqs += ">{}\n".format(i)
        for j in range(l):
            p = X_p[i, j]
            if sampling == 'max':   # only take the one with max probability
                k = np.argmax(p)
            elif sampling == 'multinomial':        # sample from multinomial
                k = np.random.choice(range(len(p)), p=p)
            aa = IDX_AA[k]
            if aa != '-':
                seqs += IDX_AA[k]
            # if nchar % 60 == 0:    # optional
            #     seqs += "\n"
        seqs += "\n"
    with open(fname, "w") as f:
        f.write(seqs)

            
class VAE(nn.Module):
    def __init__(self, **kwargs):
        super(VAE, self).__init__()

        self.seqlen = kwargs["seqlen"]
        self.n_tokens = kwargs["n_tokens"]
        self.latent_dim = kwargs["latent_dim"]
        self.enc_units = kwargs["enc_units"]

        self.encoder = nn.Sequential(
            nn.Linear(self.seqlen*self.n_tokens, self.enc_units),
            nn.ELU(),
        )
        self.mean = nn.Linear(self.enc_units, self.latent_dim)
        self.var = nn.Linear(self.enc_units, self.latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.enc_units),
            nn.ELU(),
            nn.Linear(self.enc_units, self.seqlen*self.n_tokens),
        )
        self.getprobs = nn.Softmax(dim=-1)

    def encode(self, x):
        z = self.encoder(x)
        mean = self.mean(z)
        logvar = self.var(z)
        return [mean, logvar]

    def decode(self, z):
        xhat = self.decoder(z).view(-1, self.seqlen, self.n_tokens)
        xhat = self.getprobs(xhat)
        return xhat

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps*std

    def forward(self, x, **kwargs):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return [self.decode(z), x, mean, logvar]

    def loss(self, *args, **kwargs):
        xhat = args[0]
        x = args[1]
        mean = args[2]
        logvar = args[3]        

        kl_weight = kwargs['kl_weight']

        x = x.view(-1, self.seqlen, self.n_tokens)
        # x = torch.argmax(x, -1).flatten()
        # xhat = xhat.flatten(end_dim=1)
        # recon_loss = F.cross_entropy(xhat, x.type(torch.long))
        recon_loss = F.mse_loss(x, xhat)

        kl_loss = torch.mean(-0.5*torch.sum(1 + logvar - mean**2 - logvar.exp(), dim=1), dim=0)

        loss = recon_loss + kl_weight * kl_loss

        return {'loss': loss, 'recon_loss': recon_loss, 'kl_loss': -kl_loss}

    def sample(self, num_samples, device, **kwargs):
        z = torch.randn(num_samples, self.latent_dim).to(device)
        return self.decode(z)

    def reconstruct(self, x, **kwargs):
        recon = self.forward(x)
        return recon[0]
