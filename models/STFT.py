import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import numpy as np

def hanning_window(K):
    grid = torch.arange(K).float()/K
    hanning_1d = torch.sin(grid*np.pi)**2
    hanning_1d = hanning_1d.reshape(K,1)
    return torch.matmul(hanning_1d, hanning_1d.t())

def gaussian_window(K):
    grid = torch.arange(K).float()/K
    gaussian_1d = torch.exp(-18*(grid-0.5)**2)
    gaussian_1d = gaussian_1d.reshape(K, 1)
    return torch.matmul(gaussian_1d, gaussian_1d.t())

class STFT(nn.Module):
    def __init__(self, K, window, stride, complex):
        super(STFT, self).__init__()
        assert window in ['hanning', 'gaussian', 'rectangle']
        self.K = K
        self.window = window
        if window == 'hanning':
            self.window_filter = hanning_window(K)
        elif window == 'gaussian':
            self.window_filter = gaussian_window(K)
        else: #rectangle
            self.window_filter = torch.ones(K, K)
        self.stride = stride
        self.complex = complex

    def forward(self,x):
        # x is (B,C,N,N)
        if self.window in ['hanning', 'gaussian']:
            x = F.pad(x, (self.K//2,)*4, "reflect") # (B,C,N+K,N+K)
            output = F.unfold(x, kernel_size=self.K, stride=self.stride) # (B,C*(K)**2,(N/stride+1)**2)
        else:
            output = F.unfold(x, kernel_size=self.K, stride=self.stride)  # (B,C*(K)**2,(N-K/stride+1)**2)
        output = output.reshape(output.shape[0], x.shape[1], self.K, self.K, output.shape[-1]) #(B,C,K,K,* **2) with * either 2N/K+1 or N/K
        output = output.reshape(output.shape[:4]+(int(np.sqrt(output.shape[-1])), int(np.sqrt(output.shape[-1]))))  #(B,C,K,K,*,*)
        output = output.permute(0, 1, 4, 5, 2, 3)  # (B,C,*,*,K,K)
        output = output*(self.window_filter.cuda()) # (B,C,*,*,K,K)
        output = torch.fft.fft2(output, dim=(-2, -1)) # (B,C,*,*,K,K)
        output = output.reshape(output.shape[:4]+(output.shape[-1]**2,))  # (B,C,*,*,K**2)
        output = output.permute(0,1,-1,2,3) #(B,C,K**2,*,*)
        if not self.complex:
            output = torch.cat([output[:, :, 0].real.unsqueeze(2), torch.view_as_real(output[:, :, 1:]).permute(0, 1, -1, 2, 3, 4).flatten(2, 3)], dim=2)
            #output = torch.view_as_real(output).permute(0, -1, 1, 2, 3).flatten(1, 2)
        output = output.flatten(1,2) #(B,C*K**2,*,*)

        return output
