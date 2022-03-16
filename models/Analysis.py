import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *


class Analysis(nn.Module):
    """ Computes F^T rho F x, where rho is some non-linearity. Takes care of normalization of frame atoms. """
    def __init__(self, input_type: TensorType, frame_size, non_linearity, norm_ratio, parseval, quadrature):
        """
        :param input_type:
        :param frame_size: number of atoms in the frame
        :param non_linearity: non-linearity module
        :param norm_ratio: ratio between max allowed norm and min allowed norm, disables normalization if < 1
        :param parseval: whether this module will be subject to Parseval regularization
        :param quadrature: whether this module will be subject to quadrature regularization
        """
        super(Analysis, self).__init__()

        self.input_type = input_type
        assert not self.input_type.complex
        self.output_type = self.input_type

        self.num_channels = self.input_type.num_channels
        frame = nn.Conv2d(self.num_channels, frame_size, kernel_size=1, bias=False).weight.data
        self.parseval = parseval
        self.quadrature = quadrature
        if self.parseval:
            nn.init.orthogonal_(frame)

        if 1 > norm_ratio > 0:
            raise ValueError('Normalization ratio must be either 0 or greater than 1')
        self.norm_ratio = norm_ratio
        self.mean_norm = np.sqrt(min(self.num_channels, frame_size) / frame_size)
        normalize(frame, self.norm_ratio, self.mean_norm)

        self.frame_size = frame_size
        self.frame = nn.Parameter(frame)  # (frame_size, nb_channels_in, 1, 1)
        self.non_linearity = non_linearity

    def extra_repr(self) -> str:
        return f"in_channels={self.frame.shape[1]}, frame_size={self.frame_size}, norm_ratio={self.norm_ratio}, " \
               f"parseval={self.parseval}, quadrature={self.quadrature}"

    def model_info(self) -> str:
        """ Extra information about this module: mutual coherence and atom norms. """
        frame = self.frame.reshape((self.frame_size, -1))  # (N, C)
        gram = frame @ frame.conj().t()  # (N, N)
        indices = torch.triu_indices(self.frame_size, self.frame_size, offset=1, device=gram.device)  # (2, K(K-1)/2)
        gram = gram[indices[0], indices[1]].abs()
        coherence_str = f"coherence: max {gram.max():.3f} median {gram.median():.3f}"

        norms = frame.data.norm(p=2, dim=1)
        norms_str = f"\n  - Max norm / min norm: {norms.max() / norms.min():.2f}, " \
                    f"mean norm: {norms.mean():.2f} (ref: {self.mean_norm:.2f})"

        return coherence_str + norms_str

    def forward(self, x: Tensor) -> Tensor:
        x = F.conv2d(x, self.frame)
        x = self.non_linearity(x)
        x = F.conv_transpose2d(x, self.frame)
        return x

    def on_weight_update(self):
        normalize(self.frame.data, self.norm_ratio, self.mean_norm)


class StructuredAnalysis(AbstractStructuredModule):
    """ Diagonal Analysis module on each group. The size of each sub-frame is proportional to the size of the group. """
    def __init__(self, in_channels, frame_total_size, preserve_groups, non_linearity, norm_ratio, parseval, complex):
        """
        :param in_channels: ordered dictionary, group_key -> number of input channels
        :param frame_total_size: total number of atoms in the global frame
        :param non_linearity: non-linearity module
        :param norm_ratio: ratio between max allowed norm and min allowed norm, disables normalization if < 1
        :param parseval: whether this module will be subject to Parseval regularization
        """
        total_in_channels = sum(in_channels.values())
        super().__init__(in_channels=total_in_channels)

        self.groups = in_channels
        self.frame_total_size = frame_total_size
        self.analysis_kwargs = dict(non_linearity=non_linearity, norm_ratio=norm_ratio, parseval=parseval, complex=complex)
        self.preserve_groups = preserve_groups
        self.complex = complex

        self.build()

    def build_module(self, out_key, in_channels):
        """ Builds an Analysis submodule. Frame size is proportionnal to in_channels, out_channels = in_channels. """
        if self.preserve_groups is not None and out_key in self.preserve_groups:
            if self.complex:
                return ComplexIdentity(), 2*in_channels
            else:
                return nn.Identity(), in_channels
        else:
            frame_size = (self.frame_total_size * in_channels) // self.in_channels
            if self.complex:
                out_channels = 2*in_channels
            else:
                out_channels = in_channels
            return Analysis(nb_channels_in=in_channels, frame_size=frame_size, **self.analysis_kwargs), out_channels

    def keys(self):
        """ This module does not change the keys. """
        return self.groups.keys()

    def split_in_groups(self, x):
        """ Split the input in groups, assuming the same order. """
        c = 0
        for key, num_channels in self.groups.items():
            yield key, x[:, c:c + num_channels]
            c += num_channels


def normalize(frame, norm_ratio, mean_norm):
    """ Normalize each frame vector in place.
    :param frame: (C_out, C_in, K, K) frame to normalize
    :param norm_ratio: ratio between max allowed norm and min allowed norm, disables normalization if < 1
    :param mean_norm: geometric mean of the max allowed norm and min allowed norm
    """
    if norm_ratio > 1:
        norms = frame.data.norm(p=2, dim=(1, 2, 3), keepdim=True)

        lower_bound = mean_norm / np.sqrt(norm_ratio)
        idx = norms < lower_bound
        if idx.any():
            frame.data[idx] /= norms[idx] / lower_bound

        upper_bound = mean_norm * np.sqrt(norm_ratio)
        idx = norms > upper_bound
        if idx.any():
            frame.data[idx] /= norms[idx] / upper_bound

    elif norm_ratio == 1:
        norms = frame.data.norm(p=2, dim=(1, 2, 3), keepdim=True)
        frame.data /= (norms / mean_norm)
    # else: no normalization.


class AnalysisNonLinearity(nn.Module):
    """ Applies one specified non-linearity with fixed kwargs. """
    def __init__(self, name, **kwargs):
        super().__init__()
        self.name = name
        self.non_linearity = non_linearities[name]
        self.kwargs = kwargs

    def extra_repr(self) -> str:
        return f"name={self.name}, " + ", ".join(f"{key}={value}" for key, value in self.kwargs.items())

    def forward(self, x):
        return self.non_linearity(x, **self.kwargs)


def relu(x, lambd, **kwargs):
    return F.relu(x - lambd)


def modulus(x, **kwargs):
    return torch.abs(x)


def softshrink(x, lambd, **kwargs):
    return F.relu(x - lambd) - F.relu(-x - lambd)


def softmodulus(x, lambd, **kwargs):
    """ |softshrink(x, lambd)| """
    return F.relu(x - lambd, inplace=True) + F.relu(-x - lambd, inplace=True)


def hardshrink(x, lambd, **kwargs):
    return x*((torch.abs(x) > lambd).float())


def heaviside(x, lambd, **kwargs):
    return 2 * d(x - lambd > 0).float() - 1


def identity(x, **kwargs):
    return x


def gate(x, lambd, delta, **kwargs):
    return (x - lambd > 0).float() - (x - lambd > delta).float()


def triang(x, lambd, delta, **kwargs):
    return F.relu(1. - torch.abs(x - lambd)/delta, inplace=True)


def sigmo(x, lambd, delta, **kwargs):
    return F.sigmoid((x-lambd)/delta)


def sigmo_deriv(x, lambd, delta, **kwargs):
    return torch.mul(sigmo(x, lambd, delta), 1 - sigmo(x, lambd, delta))


non_linearities = dict(
    relu=relu, modulus=modulus, softshrink=softshrink, softmodulus=softmodulus, hardshrink=hardshrink,
    heaviside=heaviside, id=identity, gate=gate, triang=triang, sigmo=sigmo, sigmo_deriv=sigmo_deriv,
)