import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def DCT_I(K, orthogonal=False):
    grid = torch.arange(1, K-1).float() / (K - 1)  # (K-2,)
    dct_I_1d = torch.zeros(K, K)
    for k in range(K):
        if orthogonal:
            dct_I_1d[k, 0] = 1/np.sqrt(2)
            dct_I_1d[k, -1] = (-1)**k/np.sqrt(2)
        else:
            dct_I_1d[k, 0] = 0.5
            dct_I_1d[k, -1] = 0.5 * (-1) ** k
        dct_I_1d[k, 1:-1] = torch.cos(grid*np.pi*k)

    if orthogonal:
        dct_I_1d[0] /= np.sqrt(2)
        dct_I_1d[-1] /= np.sqrt(2)

    return torch.einsum('ac,bd->abcd', dct_I_1d, dct_I_1d) # (K,K,K,K)

def DCT_II(K, orthogonal=False):
    grid = (torch.arange(K).float() + 0.5) / K  # (K,)
    dct_II_1d = torch.zeros(K, K)
    for k in range(K):
        dct_II_1d[k] = torch.cos(grid*np.pi*k)

    if orthogonal:
        dct_II_1d[0] /= np.sqrt(2)

    return torch.einsum('ac,bd->abcd', dct_II_1d, dct_II_1d) # (K,K,K,K)

def DCT_III(K, orthogonal=False):
    grid = torch.arange(1, K).float() / K  # (K-1)
    dct_III_1d = torch.zeros(K, K)
    for k in range(K):
        if orthogonal:
            dct_III_1d[k, 0] = 1/np.sqrt(2)
        else:
            dct_III_1d[k, 0] = 0.5
        dct_III_1d[k, 1:] = torch.cos(grid*np.pi*(k+0.5))

    return torch.einsum('ac,bd->abcd', dct_III_1d, dct_III_1d) # (K,K,K,K)

def DCT_IV(K, orthogonal=True):   # orthogonal is not used, just to be consistent with other DCTs
    grid = (torch.arange(K).float() + 0.5) / K  # (K,)
    dct_IV_1d = torch.zeros(K, K)
    for k in range(K):
        dct_IV_1d[k] = torch.cos(grid*np.pi*(k+0.5))

    return torch.einsum('ac,bd->abcd', dct_IV_1d, dct_IV_1d) # (K,K,K,K)


class DCT(nn.Module):
    def __init__(self, K, type, stride, orthogonal=False):
        super(DCT, self).__init__()
        assert type in ['I', 'II', 'III', 'IV']
        dct_dict = {'I': DCT_I, 'II': DCT_II, 'III': DCT_III, 'IV': DCT_IV}
        self.K = K
        self.dct = ((dct_dict[type](K, orthogonal=orthogonal)).flatten(0, 1)).unsqueeze(1)  #(K**2,1,K,K)
        self.stride = stride

    def forward(self, x):
        # x is (B,C,N,N)
        x = F.pad(x, (self.K//2,)*4, "reflect")  # (B,C,N+K,N+K)
        output = x.flatten(0, 1).unsqueeze(1)  # (BC, 1, N+K,N+K)
        output = torch.nn.functional.conv2d(output, self.dct.cuda(), stride=self.stride)  # (BC, K**2, (N/stride+1), (N/stride+1))
        output = output.reshape((x.shape[0], x.shape[1])+output.shape[1:]) #(B,C,K**2,(N/stride+1), (N/stride+1)))
        output = output.flatten(1,2) #(B,C*K**2,(N/stride+1), (N/stride+1)))

        return output
