import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *


class Classifier(nn.Module):
    def __init__(self, input_type: TensorType, nb_classes, avg_ker_size=1, avgpool=False):
        super().__init__()

        self.input_type = input_type
        self.num_input_channels = (1 + self.input_type.complex) * self.input_type.num_channels
        spatial_shape = np.array(self.input_type.spatial_shape)

        self.bn = nn.BatchNorm2d(self.num_input_channels)

        self.avg_ker_size = avg_ker_size
        self.avgpool = avgpool
        if self.avgpool:
            spatial_shape = np.array([1, 1])
        elif self.avg_ker_size > 1:
            spatial_shape = spatial_shape - avg_ker_size + 1

        num_features = self.num_input_channels * np.prod(spatial_shape)
        self.classifier = nn.Linear(num_features, nb_classes)

        self.output_type = TensorType(num_channels=nb_classes, spatial_shape=None, complex=False)

    def forward(self, x):
        if self.input_type.complex:
            x = torch.view_as_real(x).permute(0, 1, -1, 2, 3).flatten(1, 2)

        x = self.bn(x)

        if self.avgpool:
            x = F.adaptive_avg_pool2d(x, (1, 1))
        elif self.avg_ker_size > 1:
            x = F.avg_pool2d(x, self.avg_ker_size, stride=1)

        x = x.reshape((x.shape[0], -1))
        output = self.classifier(x)
        return output

    def equivalent_classifier(self):
        """ Returns the equivalent classifier as a (K, CN^2) weight matrix and (K,) bias """
        shape = (self.nb_channels_in, self.n_space, self.n_space)
        size = self.nb_channels_in * self.n_space ** 2
        device = next(self.parameters()).device

        bias = self(torch.zeros((1,) + shape, device=device))[0]  # (K,)
        weight = (self(torch.eye(size, device=device).reshape((size,) + shape)) - bias).t()  # (K, CN^2)

        return weight, bias
