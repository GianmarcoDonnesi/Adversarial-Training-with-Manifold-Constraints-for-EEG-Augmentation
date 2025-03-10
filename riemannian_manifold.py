import numpy as np
from sklearn.base import TransformerMixin
from pyriemann.tangentspace import TangentSpace
from numpy.linalg import eigh, pinv

class Riemann(TransformerMixin):

    def __init__(self, n = 5):

        self.n = n
        self.ts = [TangentSpace(metric = 'riemann') for _ in range(self.n)]

    def transform(self, X):

        n1, n2, p, _ = X.shape
        ret = np.empty((n1, n2, p*(p+1)//2))
        for i in range(n2):
            ret[:, i, :] = self.ts[i].transform(X[:, i, :, :])
        return ret.reshape(n1, -1)

class ProjectionCS(TransformerMixin):

    def __init__(self, n_rank = 24, s=1, r = 1e-5):

        self.s = s
        self.n_rank = n_rank
        self.r = r

    def fit(self, X):

        _, n2, _, _ = X.shape
        self.f = []
        self.p = []
        for i in range(n2):
            c = X[:, i]
            x = c.mean(axis=0)
            e_va, e_ve = eigh(x)
            ix = np.argsort(np.abs(e_va))[::-1]
            e = e_ve[:, ix]
            e = e[:, :self.n_rank].T
            self.f.append(e)
            self.p.append(pinv(e).T)
        return self

    def transform(self, X):

        n1, n2, _, _ = X.shape
        ret = np.empty((n1, n2, self.n_rank, self.n_rank))
        x2 = self.s * X
        for i in range(n2):
            f = self.f[i]
            for j in range(n1):
                ret[j, i] = f @ x2[j, i] @ f.T
                ret[j, i] += self.r * np.eye(self.n_rank)
        return ret
