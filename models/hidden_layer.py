import torch.nn as nn
import torch.nn.functional as F

class hidden_layer(nn.Module):
    def __init__(self, dictionary_size, dict_kernel_size, dict_stride, nb_channels_in):
        super(hidden_layer, self).__init__()
        self.dictionary_size = dictionary_size
        self.dict_kernel_size = dict_kernel_size
        self.dict_stride = dict_stride
        self.dict = nn.Conv2d(nb_channels_in, dictionary_size, kernel_size=dict_kernel_size, stride=dict_stride)

    def forward(self, x):
        # x is (B,C,N,N)
        #x = F.pad(x, (self.dict_kernel_size//2,)*4, "reflect") # (B,C,N+K,N+K)
        output = F.relu(self.dict(x))

        return output
