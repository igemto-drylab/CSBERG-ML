import warnings
warnings.filterwarnings("ignore")

import numpy as np
import tqdm.notebook as tq
from util import AA, AA_IDX, BLOSUM


"""
This module contains coding for training a Gaussian Process Regression
model on the Sarkisyan (2016) data set.
"""

class SequenceGP(object):
    
    def __init__(self, load=False, X_train=None, y_train=None, kernel=None,
                 length_scale=1, homo_noise=0.1, load_prefix="gfp_gp", 
                 k_beta=0.1, c=1, d=2):
        if load:
            self.load(prefix=load_prefix)
        else:
            assert X_train is not None and y_train is not None
            self.X_ = np.copy(X_train)
            self.y_ = np.copy(y_train).reshape((y_train.shape[0], 1))
            self.N_ = self.X_.shape[0]
            self.params_ = np.array([homo_noise, k_beta, c, d])
            self.K_ = None
            self.Kinv_ = None
            self._kernel = kernel
    
    def _fill_K(self, print_every=100):
        self.K_ = np.zeros((self.N_, self.N_))
        total = self.N_ * (self.N_+1) / 2
        homo_noise = self.params_[0]
        for i in tq.tqdm(range(self.N_)):
            for j in range(i, self.N_):
                kij = self._kernel(self.X_[i], self.X_[j])
                if i == j:
                    kij += homo_noise
                self.K_[i, j] = kij
                self.K_[j, i] = kij
                
        
    def _invert_K(self):
        print("Inverting K...")
        self.Kinv_ = np.linalg.inv(self.K_)
        print("Done inverting K.")
        
    def build(self, print_every=100):
        self._fill_K(print_every=print_every)
        self._invert_K()
        
    def predict(self, Xstar):
        M = len(Xstar)
        Kstar = np.zeros((M, self.N_))
        for i in range(M):
            for j in range(self.N_):
                kij = self._kernel(Xstar[i], self.X_[j])
                Kstar[i, j] = kij
        Kstarstar = np.zeros((M, M))
        for i in range(M):
            for j in range(M):
                kij = self._kernel(Xstar[i], Xstar[j])
                Kstarstar[i, j] = kij
        sigma_star = Kstarstar - np.linalg.multi_dot([Kstar, self.Kinv_, Kstar.T])
        mu_star = np.linalg.multi_dot([Kstar, self.Kinv_, self.y_])
        
        return mu_star, sigma_star
        
    def save(self, prefix = "gfp_gp"):
        np.save(prefix + "X.npy", self.X_)
        np.save(prefix + "y.npy", self.y_)
        np.save(prefix + "K.npy", self.K_)
        np.save(prefix + "Kinv.npy", self.Kinv_)
        np.save(prefix + "params.npy", self.params_)
        
    def load(self, prefix="gfp_gp"):
        self.X_ = np.load(prefix + "X.npy")
        self.y_ = np.load(prefix + "y.npy")
        self.K_ = np.load(prefix + "K.npy")
        self.Kinv_ = np.load(prefix + "Kinv.npy")
        self.params_ = np.load(prefix + "params.npy")
        self.N_ = self.X_.shape[0]
