""" Ideally all standardizations/normalizations should be here. """

import torch
import torch.nn as nn
from utils import *


class Normalization(nn.Module):
    """ Sets the norm along specified dim(s) to 1. """
    def __init__(self, input_type: TensorType, dim=1, p=2):
        """
        :param input_type:
        :param dim: int or tuple of ints, dimensions along which to compute the norm
        :param p: norm parameter
        """
        super().__init__()

        self.input_type = input_type
        self.output_type = self.input_type

        self.dim = dim
        self.p = p

    def extra_repr(self) -> str:
        return f"dim={self.dim}, p={self.p}"

    def forward(self, x: Tensor) -> Tensor:
        x = x / x.norm(p=self.p, dim=self.dim, keepdim=True)
        return x


class Standardization(nn.Module):
    """ Standardizes along specified dim(s). """
    def __init__(self, input_type: TensorType, dim=1, shape=None, remove_mean=True, eps=1e-05, momentum=0.1):
        """
        :param input_type:
        :param dim: int or tuple of ints, dimensions to preserve
        :param shape: int or tuple of ints (same as `dim`), sizes of the dimensions to preserve
        :param remove_mean: whether to remove mean (default) or simply normalize energy to 1
        :param eps: regularization of variance
        :param momentum: used for running stats
        """
        super().__init__()

        self.input_type = input_type

        self.dim = to_tuple(dim)  # Non-negative and strictly increasing
        assert self.dim[0] >= 0 and all(self.dim[i + 1] > self.dim[i] for i in range(len(self.dim) - 1))

        if shape is None:
            assert self.dim == (1,)
            self.shape = (self.input_type.num_channels,)
        else:
            self.shape = to_tuple(shape)  # Same length as self.dim
            assert len(self.shape) == len(self.dim)

        self.complex = self.input_type.complex
        self.output_type = self.input_type

        self.remove_mean = remove_mean
        if self.complex:
            mean = torch.view_as_real(torch.zeros(self.shape, dtype=torch.complex64))
        else:
            mean = torch.zeros(self.shape)
        self.register_buffer("mean", mean)  # complex or real
        self.register_buffer("var", torch.ones(self.shape))  # real

        self.eps = eps
        self.momentum = momentum

    def extra_repr(self) -> str:
        return f"dim={self.dim}, shape={self.shape}, complex={self.complex}, remove_mean={self.remove_mean}"

    def forward(self, x: Tensor) -> Tensor:
        # issue with DataParallel for the moment...
        def get_mean():
            mean = self.mean
            if self.complex:
                if mean.storage_offset() % 2 == 0:
                    mean = torch.view_as_complex(mean)
                else:  # Workaround for bug in DataParallel/view_as_complex
                    mean = torch.complex(mean[:, 0], mean[:, 1])
            return mean

        def set_mean(mean):
            if self.complex:
                mean = torch.view_as_real(mean)
            self.mean.copy_(mean)

        index = tuple(Ellipsis if i in self.dim else None for i in range(x.ndim))

        if self.training:
            # Use batch statistics for whitening during training.
            avg_dims = tuple(i for i in range(x.ndim) if i not in self.dim)

            if self.remove_mean:
                mean = torch.mean(x, avg_dims)
            else:
                mean = get_mean()  # zeroes

            # torch.var does not work with autograd on complex input, hence we manually compute the variance.
            diff = x - mean[index]
            if torch.is_complex(diff):
                diff_sq = torch.real(diff * diff.conj())
            else:
                diff_sq = diff * diff
            var = torch.mean(diff_sq, avg_dims)  # var is real even though x may be complex.
            num_samples = x.numel() // var.numel()
            unbiased_var = num_samples / (num_samples - 1) * var

            # Update statistics. copy_ necessary for DataParallel.
            with torch.no_grad():
                set_mean(self.momentum * mean + (1 - self.momentum) * get_mean())
                # Update var with unbiased var.
                self.var.copy_(self.momentum * unbiased_var + (1 - self.momentum) * self.var)
        else:
            # Use computed statistics for whitening during validation.
            var, mean = self.var, get_mean()

        # Whiten x.
        x = (x - mean[index]) * torch.rsqrt(self.eps + var[index])
        return x
