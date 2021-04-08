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
from typing import Union, List, Tuple, Sequence, Dict, Any, Optional, Collection
from copy import copy
from pathlib import Path
import pickle as pkl
import logging
import random
import lmdb
from scipy.spatial.distance import pdist, squareform


plt.rcParams['figure.dpi'] = 300


AA = list("-ACDEFGHIKLMNPQRSTVWY")
AA_IDX = {AA[i]:i for i in range(len(AA))}
IDX_AA = {i:AA[i].upper() for i in range(len(AA))}

# from CbAS' util.py
BLOSUM = np.array([
[3.9029,0.6127,0.5883,0.5446,0.8680,0.7568,0.7413,1.0569,0.5694,0.6325,0.6019,0.7754,0.7232,0.4649,0.7541,1.4721,0.9844,0.4165,0.5426,0.9365],
[0.6127,6.6656,0.8586,0.5732,0.3089,1.4058,0.9608,0.4500,0.9170,0.3548,0.4739,2.0768,0.6226,0.3807,0.4815,0.7672,0.6778,0.3951,0.5560,0.4201],
[0.5883,0.8586,7.0941,1.5539,0.3978,1.0006,0.9113,0.8637,1.2220,0.3279,0.3100,0.9398,0.4745,0.3543,0.4999,1.2315,0.9842,0.2778,0.4860,0.3690],
[0.5446,0.5732,1.5539,7.3979,0.3015,0.8971,1.6878,0.6343,0.6786,0.3390,0.2866,0.7841,0.3465,0.2990,0.5987,0.9135,0.6948,0.2321,0.3457,0.3365],
[0.8680,0.3089,0.3978,0.3015,19.5766,0.3658,0.2859,0.4204,0.3550,0.6535,0.6423,0.3491,0.6114,0.4390,0.3796,0.7384,0.7406,0.4500,0.4342,0.7558],
[0.7568,1.4058,1.0006,0.8971,0.3658,6.2444,1.9017,0.5386,1.1680,0.3829,0.4773,1.5543,0.8643,0.3340,0.6413,0.9656,0.7913,0.5094,0.6111,0.4668],
[0.7413,0.9608,0.9113,1.6878,0.2859,1.9017,5.4695,0.4813,0.9600,0.3305,0.3729,1.3083,0.5003,0.3307,0.6792,0.9504,0.7414,0.3743,0.4965,0.4289],
[1.0569,0.4500,0.8637,0.6343,0.4204,0.5386,0.4813,6.8763,0.4930,0.2750,0.2845,0.5889,0.3955,0.3406,0.4774,0.9036,0.5793,0.4217,0.3487,0.3370],
[0.5694,0.9170,1.2220,0.6786,0.3550,1.1680,0.9600,0.4930,13.5060,0.3263,0.3807,0.7789,0.5841,0.6520,0.4729,0.7367,0.5575,0.4441,1.7979,0.3394],
[0.6325,0.3548,0.3279,0.3390,0.6535,0.3829,0.3305,0.2750,0.3263,3.9979,1.6944,0.3964,1.4777,0.9458,0.3847,0.4432,0.7798,0.4089,0.6304,2.4175],
[0.6019,0.4739,0.3100,0.2866,0.6423,0.4773,0.3729,0.2845,0.3807,1.6944,3.7966,0.4283,1.9943,1.1546,0.3711,0.4289,0.6603,0.5680,0.6921,1.3142],
[0.7754,2.0768,0.9398,0.7841,0.3491,1.5543,1.3083,0.5889,0.7789,0.3964,0.4283,4.7643,0.6253,0.3440,0.7038,0.9319,0.7929,0.3589,0.5322,0.4565],
[0.7232,0.6226,0.4745,0.3465,0.6114,0.8643,0.5003,0.3955,0.5841,1.4777,1.9943,0.6253,6.4815,1.0044,0.4239,0.5986,0.7938,0.6103,0.7084,1.2689],
[0.4649,0.3807,0.3543,0.2990,0.4390,0.3340,0.3307,0.3406,0.6520,0.9458,1.1546,0.3440,1.0044,8.1288,0.2874,0.4400,0.4817,1.3744,2.7694,0.7451],
[0.7541,0.4815,0.4999,0.5987,0.3796,0.6413,0.6792,0.4774,0.4729,0.3847,0.3711,0.7038,0.4239,0.2874,12.8375,0.7555,0.6889,0.2818,0.3635,0.4431],
[1.4721,0.7672,1.2315,0.9135,0.7384,0.9656,0.9504,0.9036,0.7367,0.4432,0.4289,0.9319,0.5986,0.4400,0.7555,3.8428,1.6139,0.3853,0.5575,0.5652],
[0.9844,0.6778,0.9842,0.6948,0.7406,0.7913,0.7414,0.5793,0.5575,0.7798,0.6603,0.7929,0.7938,0.4817,0.6889,1.6139,4.8321,0.4309,0.5732,0.9809],
[0.4165,0.3951,0.2778,0.2321,0.4500,0.5094,0.3743,0.4217,0.4441,0.4089,0.5680,0.3589,0.6103,1.3744,0.2818,0.3853,0.4309,38.1078,2.1098,0.3745],
[0.5426,0.5560,0.4860,0.3457,0.4342,0.6111,0.4965,0.3487,1.7979,0.6304,0.6921,0.5322,0.7084,2.7694,0.3635,0.5575,0.5732,2.1098,9.8322,0.6580],
[0.9365,0.4201,0.3690,0.3365,0.7558,0.4668,0.4289,0.3370,0.3394,2.4175,1.3142,0.4565,1.2689,0.7451,0.4431,0.5652,0.9809,0.3745,0.6580,3.6922]]
)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LMDBDataset(Dataset):
    """Creates a dataset from an lmdb file.
    Args:
        data_file (Union[str, Path]): Path to lmdb file.
        in_memory (bool, optional): Whether to load the full dataset into memory.
            Default: False.
    """

    def __init__(self,
                 data_file: Union[str, Path],
                 in_memory: bool = False):

        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)

        env = lmdb.open(str(data_file), max_readers=1, readonly=True,
                        lock=False, readahead=False, meminit=False)

        with env.begin(write=False) as txn:
            num_examples = pkl.loads(txn.get(b'num_examples'))

        if in_memory:
            cache = [None] * num_examples
            self._cache = cache

        self._env = env
        self._in_memory = in_memory
        self._num_examples = num_examples

    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)

        if self._in_memory and self._cache[index] is not None:
            item = self._cache[index]
        else:
            with self._env.begin(write=False) as txn:
                item = pkl.loads(txn.get(str(index).encode()))
                if 'id' not in item:
                    item['id'] = str(index)
                if self._in_memory:
                    self._cache[index] = item
        return item



def one_hot_encode_aa(aa_str, pad=None):
    aa_str = aa_str.upper()
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
