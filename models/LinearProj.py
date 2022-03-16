import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *


class ComplexConv2d(nn.Module):
    """ Conv2D class which also works for complex input, and can initialize its weight to a unitary operator. """
    def __init__(self, input_type: TensorType, complex_weights, out_channels, kernel_size, parseval, quadrature):
        """
        :param input_type:
        :param complex_weights: whether the weights will be complex or real (None defaults to type of input)
        :param out_channels:
        :param kernel_size:
        :param parseval: whether this module will be subject to Parseval regularization
        :param quadrature: whether this module will be subject to quadrature regularization
        """
        super().__init__()

        self.input_type = input_type
        self.in_channels = self.input_type.num_channels
        self.complex_weights = complex_weights if complex_weights is not None else self.input_type.complex
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        assert self.kernel_size == 1
        self.output_type = TensorType(self.out_channels, self.input_type.spatial_shape,
                                      complex=self.input_type.complex or self.complex_weights)

        self.parseval = parseval
        self.quadrature = quadrature

        if self.complex_weights:
            shape = (out_channels, self.in_channels, kernel_size, kernel_size)
            param = unitary_init(shape)
            param = torch.view_as_real(param)
        else:
            param = nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels,
                              kernel_size=kernel_size).weight.data
            if self.parseval:
                nn.init.orthogonal_(param)
        self.param = nn.Parameter(param)

    def extra_repr(self) -> str:
        def format_complex(complex):
            return 'C' if complex else 'R'
        return f"in_channels={type_to_str(self.input_type)}, out_channels={type_to_str(self.output_type)}, " \
               f"kernel_size={self.kernel_size}, complex_weights={self.complex_weights}, parseval={self.parseval}, " \
               f"quadrature={self.quadrature}"

    def forward(self, x):
        return conv2d(x, self.param, self.output_type.complex)


class TriangularComplexConv2d(nn.Module):
    """ Efficient (batched) implementation of a convolution with block-triangular weights across channels.
    Equivalent to Sequential(TriangularModule(), DiagonalModule(ComplexConv2d)) but faster. """
    def __init__(self, input_type: SplitTensorType, complex_weights, out_channels, kernel_size, parseval, quadrature):
        """
        :param input_type: split tensor type
        :param complex_weights: whether the weights will be complex or real (None defaults to type of input)
        :param out_channels: dictionary, group key -> number of output channels
        :param kernel_size: global kernel size
        :param parseval: whether to apply Parseval regularization on the full triangular convolution matrix
        :param quadrature: whether to apply quadrature regularization on the full triangular convolution matrix
        """
        super().__init__()

        self.input_type = input_type
        self.kernel_size = kernel_size
        self.complex_weights = complex_weights if complex_weights is not None else self.input_type.complex
        self.parseval = parseval
        self.quadrature = quadrature

        self.keys = self.input_type.keys
        in_channels = 0
        submodules = {}
        for key in self.keys:
            in_channels += input_type.groups[key]
            # Parseval/Quadrature is handled by this module, hence we transmit parseval/quadrature=False to submodules.
            submodules[key] = ComplexConv2d(
                input_type=TensorType(in_channels, input_type.spatial_shape, input_type.complex),
                complex_weights=complex_weights, out_channels=out_channels[key],
                kernel_size=self.kernel_size, parseval=False, quadrature=False,
            )
        self.submodules = ModuleDict(submodules)
        self.total_in_channels = in_channels
        self.total_out_channels = sum(out_channels.values())

        self.output_type = SplitTensorType(
            groups=out_channels, spatial_shape=next(self.submodules.values().__iter__()).output_type.spatial_shape,
            complex=self.input_type.complex or self.complex_weights,
        )

    def extra_repr(self):
        def format_complex(complex):
            return 'C' if complex else 'R'
        return f"in_channels={self.total_in_channels}{format_complex(self.input_type.complex)}, " \
               f"out_channels={self.total_out_channels}{format_complex(self.output_type.complex)}, " \
               f"kernel_size={self.kernel_size}, complex_weights={self.complex_weights}, parseval={self.parseval}, " \
               f"quadrature={self.quadrature}"

    def full_weights(self):
        """ Returns the full weights, of shape (out_channels, in_channels, kernel_size, kernel_size).
        If complex weights, returns a real view with an additional last dimension of 2. """
        shape = (1, 1) + ((2,) if self.complex_weights else ())
        if len(self.submodules) == 1:  # Slight optimization (also allows in-place update).
            w = self.submodules[0].param
        else:
            w = torch.cat([torch.cat([sub.param, sub.param.new_zeros(
                (sub.param.shape[0], self.total_in_channels - sub.param.shape[1]) + shape)], dim=1)
                           for sub in self.submodules.values() if sub.param.numel() > 0], dim=0)  # deal with 0 input or output case
        return w

    def forward(self, x: SplitTensor) -> SplitTensor:
        x = x.full_view()
        w = self.full_weights()
        y = conv2d(x, w, self.output_type.complex)
        return SplitTensor(y, groups={key: self.submodules[key].out_channels for key in self.keys})


def conv2d(x, w, complex):
    """ Real or complex convolution between x (B, C, M, N, [2]) and w (K, C, H, W, [2]), handles type problems.
    x and w can be real, complex, or real with an additional trailing dimension of size 2.
    A complex convolution causes the view or cast of both x and w as complex tensors.
    Returns a real or complex tensor of size (B, K, M', N'). """
    def real_to_complex(z):
        if z.is_complex():
            return z
        elif z.ndim == 5:
            # return torch.view_as_complex(z)  # View
            return torch.complex(z[..., 0], z[..., 1])  # Temporary copy instead of view...
        elif z.ndim == 4:
            return z.type(torch.complex64)  # Cast
        else:
            assert False

    if w.shape[0] == 0:  # Stupid special case because pytorch can't handle zero-sized convolutions.
        y = x[:, 0:0]  # (B, 0, M, N), this assumes that x is the right type
        if complex:
            y = real_to_complex(y)

    else:
        if complex:
            x = real_to_complex(x)
            w = real_to_complex(w)
            conv_fn = complex_conv2d
        else:
            conv_fn = F.conv2d
        y = conv_fn(x, w)

    return y


class FwStructuredProj(AbstractStructuredModule):
    """ Specialization of the general StructuredModule to our case (Pj after F_w),
    separating orders, frequencies and angles. """
    def __init__(self, prev_out_channels, L, A, num_linear_phases, depth, complex, parseval,
                 out_channels, separate_orders, separate_freqs, separate_angles, separate_packets, throw_packets):
        """
        :param prev_out_channels: dictionary, (order, freq, angle) -> number of out channels in the previous module
        :param L: number of angles
        :param A: number of phases
        :param num_linear_phases: number of linear phases, assumed to be the first high-frequency channels
        :param depth: number of previous such modules (starts at 0)
        :param complex: generates a unitary matrix, expecting complex input and output
        :param parseval: whether this module will be subject to Parseval regularization
        :param out_channels: dictionary, (order, freq, angle) -> number of out channels of this module or "id" or slice
        :param separate_orders: whether to separate orders (triangular separation)
        :param separate_freqs: whether to separate frequencies (of last wavelet, for translation)
        :param separate_angles: whether to separate angles (of first wavelet, for rotation)
        :param separate_packets: whether to remove non-linear operations on wavelet packets
        :param throw_packets: whether to throw away linear part of high frequencies
        """
        total_prev_out_channels = sum(prev_out_channels.values())
        super().__init__(in_channels=total_prev_out_channels * (1 + A * L))

        self.prev_out_channels = prev_out_channels
        self.total_prev_out_channels = total_prev_out_channels
        self.out_channels_spec = out_channels
        self.L = L
        self.A = A
        self.num_linear_phases = num_linear_phases
        self.depth = depth

        self.separate_orders = separate_orders
        # Throwing packets implies having everything at frequency zero, hence it is as if we do not separate them.
        self.separate_freqs = separate_freqs and not throw_packets
        self.separate_angles = separate_angles
        self.separate_packets = separate_packets
        self.throw_packets = throw_packets

        self.conv_kwargs = dict(complex=complex, parseval=parseval)

        assert set(prev_out_channels.keys()) == set(self.keys(depth=depth))

        self.build()

    def build_module(self, out_key, in_channels_out_key):
        out_channels_out_key = self.out_channels_spec[out_key]

        if out_channels_out_key == 0:
            # Stupid special case because Pytorch cannot handle zero-sized convolutions.
            out_channels_out_key = idx[0:0]

        if out_channels_out_key == "id":
            # TODO: should do a slice here instead of identity, in case of triangular separation (skip connections).
            proj = nn.Identity()
            out_channels_out_key = in_channels_out_key

        elif isinstance(out_channels_out_key, slice):
            proj = ChannelSlicer(in_channels_out_key, out_channels_out_key)
            out_channels_out_key = proj.out_channels

        elif isinstance(out_channels_out_key, int):
            assert out_channels_out_key <= in_channels_out_key
            proj = ComplexConv2d(in_channels_out_key, out_channels_out_key, kernel_size=1, **self.conv_kwargs)

        else:
            assert False

        return proj, out_channels_out_key

    def keys(self, depth=None):
        """ Compute the keys after `depth` F_w modules. The keys correspond to both before and after the projections.
        depth=self.depth corresponds to before this module (and F_w), depth=self.depth + 1 after this module.
        By default, computes keys after this module (so called out-groups), for AbstractStructuredModule.
        """
        if depth is None:
            depth = self.depth + 1

        # Possible orders: all if separated, only zero otherwise.
        orders = [0]
        if self.separate_orders:
            orders += list(range(1, depth + 1))
        for order in orders:

            # Possible frequencies: always 0, -1 and 1..L requires to separate frequencies.
            # In this case, there is a limit on order (and hence depth).
            freqs = []
            if (self.separate_freqs and order <= depth - 2) or (self.separate_packets and order <= depth - 1):
                freqs += [-1]
            freqs += [0]
            if self.separate_freqs and order <= depth - 1:
                freqs += list(range(1, 1 + self.L))
            for freq in freqs:

                # Compute possible angles, which is a bit convoluted.
                angles = []
                # Angle 0 means angles are not separated, there hasn't been a high-pass filter yet, or wavelet packet.
                if not self.separate_angles or (freq == 0 and order == 0) or freq == -1:
                    angles += [0]
                # Add other angles except in some special cases (wavelet packets, no separation or no high-pass filter).
                if self.separate_angles and freq != -1 and depth >= 1 and \
                        not (self.separate_orders and order == 0 and
                             (self.separate_freqs or self.throw_packets) and freq == 0):
                    if freq > 0 and {True: order == 0, False: depth == 1}[self.separate_orders]:
                        angles += [freq]  # Only one high-pass filter, so we must have angle = freq.
                    else:
                        angles += list(range(1, 1 + self.L))
                for angle in angles:

                    yield order, freq, angle

    def split_in_groups(self, x):
        # x is (B, C, 1 + AL)
        x = channel_reshape(x, (self.total_prev_out_channels, 1 + self.A * self.L))  # (B, C_prev, 1 + LA, H, W)

        c = 0
        for prev_order, prev_freq, prev_angle in self.keys(depth=self.depth):
            num_channels = self.prev_out_channels[prev_order, prev_freq, prev_angle]
            x_prev = x[:, c:c + num_channels]  # (B, num_channels, 1 + LA, H, W)
            c += num_channels

            if prev_freq != 0:  # Wavelet packet, throw away the non-linear part.
                yield (prev_order, -1, 0), x_prev[:, :, :1 + self.num_linear_phases * self.L]

            else:
                x_prev_low = x_prev[:, :, 0]  # (B, num_channels, H, W)
                yield (prev_order, 0, prev_angle), x_prev_low

                x_prev_high = channel_reshape(x_prev[:, :, 1:], (num_channels, self.A, self.L))
                # (B, num_channels, A, L, H, W)

                if self.separate_freqs or self.separate_angles:
                    for mid_angle in range(self.L):
                        # Compute new frequency key (actually mixed with angle) and initialize angle (if first wavelet).
                        if prev_angle == 0:
                            # This means that up until the last F_w, there were only low frequency filters.
                            new_freq = 1 + mid_angle
                            new_angle = 1 + mid_angle
                        else:
                            new_freq = 1 + (mid_angle - (prev_angle - 1)) % self.L
                            new_angle = prev_angle

                        # TODO: Should separate linear phases because of pi-rotation which conjugates?
                        if not self.throw_packets:
                            yield (prev_order, new_freq, new_angle), x_prev_high[:, :, :self.num_linear_phases, mid_angle]
                        # TODO: This assumes non-linearity is modulus (frequency zero).
                        yield (prev_order + 1, 0, new_angle), x_prev_high[:, :, self.num_linear_phases:, mid_angle]

                # Faster, optimized versions of the above
                elif self.separate_orders or self.separate_packets:
                    if not self.throw_packets:
                        yield (prev_order, -1, 0), x_prev_high[:, :, :self.num_linear_phases]
                    yield (prev_order + 1, 0, 0), x_prev_high[:, :, self.num_linear_phases:]
                else:
                    yield (0, 0, 0), x_prev_high[:, :, (self.num_linear_phases if self.throw_packets else 0):]

    def out_keys(self, in_key):
        order, freq, angle = in_key

        # Map keys back to default value if we don't separate them.
        if not self.separate_orders:
            order = 0
        if not self.separate_freqs and not (self.separate_packets and freq == -1):
            freq = 0
        if not self.separate_angles:
            angle = 0

        orders = [order]
        if self.separate_orders:
            # Add higher orders (triangular separation / skip connections).
            # Special case: if we separate angles, (d, 0, 0) means d=0, this is the lowest frequency.
            # There is no use in adding higher orders because there are no coefficients like this for d > 0.
            if not (self.separate_angles and freq == 0 and angle == 0):
                if freq == 0:
                    max_order = self.depth + 1
                elif freq > 0:
                    # Final convolution with psi_la, hence no final non-linearity,
                    # and thus the maximum order is one less of its normal value.
                    max_order = self.depth
                else:  # freq == -1
                    # The two last convolutions are psi_la and then some other wavelet,
                    # which means that the maximum order is two less of its normal value.
                    max_order = self.depth if self.separate_packets else self.depth - 1

                orders += list(range(order + 1, max_order + 1))

        return [(d, freq, angle) for d in orders]

    def equivalent_proj(self, device):
        """ Returns the equivalent projector, of shape (C_out, C_in). """
        x = torch.eye(self.in_channels, device=device)[..., None, None]  # (C_in, C_in, 1, 1)
        y = self(x)  # (C_in, C_out, 1, 1)
        return y[..., 0, 0].t()  # (C_out, C_in)
