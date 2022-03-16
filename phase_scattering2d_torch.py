import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *


def filter_bank(P, Q, J, scales_per_octave, L, full_angles, factorize_filters=False, i=None):
    """
    Compute compactly supported Morlet filters in the spatial domain.
    ----------
    P, Q : int
        spatial size of the filters (need to match the one of the input for Fourier, but can be smaller for spatial)
    J : int
        logscale of the scattering
    scales_per_octave: int, optional
        number of scales per octave
    L : int
        number of angles used for the wavelet transform
    full_angles : bool
        whether to have angles ranging from 0 to pi or 0 to 2pi (knowing that psi_{theta + pi} = bar{psi_theta})
    spatial : bool
        whether to return the filters in the spatial or Fourier domain
    Returns
    -------
    filters : numpy array of shape (JSL + 1, P, Q) and dtype complex64 (if spatial) owr float32 (if Fourier),
        containing the filters in the specified domain.
        The order is the following: [psi(j=0,theta=0..L-1), ..., psi(j=J-1,theta=0..L-1), phi(j=J)].
    Notes
    -----
    The design of the filters is optimized for the value L = 8.
    """
    filters = []

    def add_filter(filter_fn, **kwargs):
        filter_signal = filter_fn(P=P, Q=Q, **kwargs)  # (P, Q), complex64
        filter_signal = np.real(np.fft.fft2(filter_signal)).astype(np.float32)  # (P, Q), float32 (filters are real in the Fourier domain)
        filters.append(filter_signal)

    if L > 0:
        if full_angles:
            max_angle = 2 * np.pi
            angles_to_pi = L / 2
        else:
            max_angle = np.pi
            angles_to_pi = L
        slant = 4.0 / angles_to_pi

    if factorize_filters:
        assert i==0 or i==1
        if i==0:  # psi_1
            for theta in range(L):
                add_filter(morlet_2d, sigma=0.8, theta=(int(L - L / 2 - 1) - theta) * max_angle / L,
                           xi=3.0 / 4.0 * np.pi, slant=slant)
        else:  # i == 1, to build psi_3/2 from phi_1/2
            for theta in range(L):
                add_filter(morlet_2d, sigma=0.8 * np.sqrt(3/2), theta=(int(L - L / 2 - 1) - theta) * max_angle / L,
                           xi= np.pi / np.sqrt(2), slant=np.sqrt(3*slant**2/(4-slant**2)))
        add_filter(gabor_2d, sigma=0.8 * 2 ** (-1/2), theta=0, xi=0)  #phi_1/2
    else:
        for j in np.arange(0, J, 1 / scales_per_octave):
            for theta in range(L):
                add_filter(morlet_2d, sigma=0.8 * 2 ** j, theta=(int(L - L / 2 - 1) - theta) * max_angle / L,
                           xi=3.0 / 4.0 * np.pi / 2 ** j, slant=slant)
        add_filter(gabor_2d, sigma=0.8 * 2 ** (J - 1), theta=0, xi=0)

    return np.stack(filters, axis=0)  # (JL + 1, P, Q)


def morlet_2d(P, Q, sigma, theta, xi, slant=0.5, offset_x=0, offset_y=0, periodize=True):
    """
    Computes a 2D Morlet filter.
    A Morlet filter is the sum of a Gabor filter and a low-pass filter
    to ensure that the sum has exactly zero mean in the temporal domain.
    It is defined by the following formula in space:
    psi(u) = g_{sigma}(u) (e^(i xi^T u) - beta)
    where g_{sigma} is a Gaussian envelope, xi is a frequency and beta is
    the cancelling parameter.

    Parameters
    ----------
    P, Q : int
        spatial size of the filter
    sigma : float
        bandwidth parameter
    xi : float
        central frequency (in [0, 1])
    theta : float
        angle in [0, pi]
    slant : float, optional
        parameter which guides the ellipsoidal shape of the morlet
    offset_x, offset_y : int, optional
        offsets by which the signal starts
    periodize: bool, optional
        whether to periodize the signal by summing its translations

    Returns
    -------
    morlet : ndarray
        numpy array of size (P, Q) of dtype complex64, containing the filter in the spatial domain
    """
    wv = gabor_2d(P, Q, sigma, theta, xi, slant, offset_x, offset_y, periodize)
    wv_modulus = gabor_2d(P, Q, sigma, theta, 0, slant, offset_x, offset_y, periodize)
    K = np.sum(wv) / np.sum(wv_modulus)

    mor = wv - K * wv_modulus
    return mor


def gabor_2d(P, Q, sigma, theta, xi, slant=1.0, offset_x=0, offset_y=0, periodize=True):
    """
    Computes a 2D Gabor filter.
    A Gabor filter is defined by the following formula in space:
    psi(u) = g_{sigma}(u) e^(i xi^T u)
    where g_{sigma} is a Gaussian envelope and xi is a frequency.

    Parameters
    ----------
    P, Q : int
        spatial size of the filter
    sigma : float
        bandwidth parameter
    xi : float
        central frequency (in [0, 1])
    theta : float
        angle in [0, pi]
    slant : float, optional
        parameter which guides the ellipsoidal shape of the morlet
    offset_x, offset_y : int, optional
        offsets by which the signal starts
    periodize: bool, optional
        whether to periodize the signal by summing its translations

    Returns
    -------
    gabor : ndarray
        numpy array of size (P, Q) of dtype complex64, containing the filter in the spatial domain
    """
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], np.float32)
    R_inv = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]], np.float32)
    D = np.array([[1, 0], [0, slant * slant]])
    curv = np.dot(R, np.dot(D, R_inv)) / (2 * sigma * sigma)

    gab = np.zeros((P, Q), np.complex64)
    foldings = [-2, -1, 0, 1, 2] if periodize else [0]
    for ex in foldings:
        for ey in foldings:
            [xx, yy] = np.mgrid[offset_x + ex * P:offset_x + (1 + ex) * P, offset_y + ey * Q:offset_y + (1 + ey) * Q]
            arg = -(curv[0, 0] * np.multiply(xx, xx) + (curv[0, 1] + curv[1, 0]) * np.multiply(xx, yy) + curv[
                1, 1] * np.multiply(yy, yy)) + 1.j * (xx * xi * np.cos(theta) + yy * xi * np.sin(theta))
            gab += np.exp(arg)

    norm_factor = (2 * 3.1415 * sigma * sigma / slant)
    gab /= norm_factor

    return gab.astype(np.complex64)


def compute_padding(M, N, J):
    """
         Precomputes the future padded size. If 2^J=M or 2^J=N,
         border effects are unavoidable in this case, and it is
         likely that the input has either a compact support,
         either is periodic.
         Parameters
         ----------
         M, N : int
             input size
         Returns
         -------
         M, N : int
             padded size
    """
    M_padded = ((M + 2 ** J) // 2 ** J + 1) * 2 ** J
    N_padded = ((N + 2 ** J) // 2 ** J + 1) * 2 ** J

    return M_padded, N_padded


class Pad(object):
    def __init__(self, pad_size, input_size):
        """Padding which allows to simultaneously pad in a reflection fashion.

            Parameters
            ----------
            pad_size : list of 4 integers
                Size of padding to apply [top, bottom, left, right].
            input_size : list of 2 integers
                size of the original signal [height, width].

        """
        self.pad_size = pad_size
        self.input_size = input_size

        self.build()

    def build(self):
        """Builds the padding module.

            Attributes
            ----------
            padding_module : ReflectionPad2d
                Pads the input tensor using the reflection of the input
                boundary.

        """
        pad_size_tmp = list(self.pad_size)

        # This handles the case where the padding is equal to the image size
        if pad_size_tmp[0] == self.input_size[0]:
            pad_size_tmp[0] -= 1
            pad_size_tmp[1] -= 1
        if pad_size_tmp[2] == self.input_size[1]:
            pad_size_tmp[2] -= 1
            pad_size_tmp[3] -= 1
        # Pytorch expects its padding as [left, right, top, bottom]
        self.padding_module = nn.ReflectionPad2d([pad_size_tmp[2], pad_size_tmp[3],
                                                  pad_size_tmp[0], pad_size_tmp[1]])

    def __call__(self, x):
        """Applies padding.

            Parameters
            ----------
            x : tensor
                Real or complex tensor input to be padded.

            Returns
            -------
            output : tensor
                Real of complex torch tensor that has been padded.

        """
        x = self.padding_module(x)

        # Note: PyTorch is not effective to pad signals of size N-1 with N
        # elements, thus we had to add this fix.
        if self.pad_size[0] == self.input_size[0]:
            x = torch.cat([x[:, :, 1, :].unsqueeze(2), x, x[:, :, x.shape[2] - 2, :].unsqueeze(2)], 2)
        if self.pad_size[2] == self.input_size[1]:
            x = torch.cat([x[:, :, :, 1].unsqueeze(3), x, x[:, :, :, x.shape[3] - 2].unsqueeze(3)], 3)

        return x


def ignore_nan_inf_gradients_hook(grad):
    if torch.isnan(grad).any():
        grad[torch.isnan(grad)] = 0.

    if torch.isinf(grad).any():
        grad[torch.isinf(grad)] = 0.

    return grad


class Scattering2D(nn.Module):
    """ Batched Scattering implementation. Returns a dict with two keys, `phi` and `psi`. """
    def __init__(self, input_type: SplitTensorType, scales_per_octave, L, full_angles, separate_freqs,
                 factorize_filters=False, i=None):
        """
        :param input_type:
        :param scales_per_octave: number of scales per octave (geometrically spaced every 2 ** (1 / scales_per_octave))
        :param L: number of angles
        :param full_angles: whether to take angles in [0, pi] or [0, 2pi]
        :param separate_freqs: whether to introduce different groups for each frequency
        """
        super().__init__()

        self.input_type = input_type
        self.total_input_channels = sum(self.input_type.groups.values())
        self.M, self.N = self.input_type.spatial_shape
        self.M_padded, self.N_padded = compute_padding(self.M, self.N, J=1)
        self.scales_per_octave = scales_per_octave
        self.L = L
        self.full_angles = full_angles
        self.separate_freqs = separate_freqs

        filters = filter_bank(
            P=self.M_padded, Q=self.N_padded, J=1, scales_per_octave=scales_per_octave, L=L, full_angles=full_angles,
            factorize_filters=factorize_filters, i=i)  # (SL + 1, M, N) float32 ndarray
        self.subsample = (not factorize_filters) or (factorize_filters and i == 1)

        self.channels_factor = self.scales_per_octave * self.L
        if self.subsample:
            self.output_spatial_shape = (self.M_padded // 2 - 2, self.N_padded // 2 - 2)
        else:
            self.output_spatial_shape = (self.M_padded - 4, self.N_padded - 4)

        self.register_buffer('phis', torch.from_numpy(filters[-1]))  # (M, N) real
        self.register_buffer('psis', torch.from_numpy(filters[:-1]))  # (SL, M, N) real

        self.pad = Pad([(self.M_padded - self.M) // 2, (self.M_padded - self.M + 1) // 2,
                        (self.N_padded - self.N) // 2, (self.N_padded - self.N + 1) // 2], [self.M, self.N])

        self.output_type = infer_output_type(self, self.input_type)

    def extra_repr(self) -> str:
        full_angles = "(full)" if self.full_angles else ""
        spatial = f"spatial=({self.M},{self.N}) to ({self.output_spatial_shape[0]},{self.output_spatial_shape[1]})"
        input = f"input_channels={type_to_str(self.input_type.tensor_type())}"
        phi = f"phi_channels={type_to_str(self.output_type['phi'])}"
        if self.L > 0:
            phi = f"{phi}, psi_channels={type_to_str(self.output_type['psi'])}"
        return f"{input}, S={self.scales_per_octave}, L={self.L}{full_angles}, {spatial}, {phi}"

    def forward(self, x: SplitTensor) -> Dict[str, SplitTensor]:
        """ (B, C, M, N) to (B, (SL)C/C(SL), M, N) complex. """
        return phase_scattering2d_batch(x_split=x, pad=self.pad, phi=self.phis, psi=self.psis,
                                        separate_freqs=self.separate_freqs, subsample=self.subsample)


def phase_scattering2d_batch(x_split: SplitTensor, pad, phi, psi, separate_freqs, subsample) -> Dict[str, SplitTensor]:
    """
    :param x_split: full view is (B, C, M, N), real or complex
    :param pad: padding module
    :param phi: (M, N) real, phi filter in Fourier
    :param psi: (JSL, M, N) real, psi filters in Fourier
    :param separate_freqs: whether to introduce different groups for each frequency
    :return: phi: (B, C, M//2, N//2) real or complex, psi: (B, SLC/CSL, M//2, N//2) complex (change frequency keys)
    """
    def unpad(x, subsample=True):  # x is (B,C,M,N)
        if subsample:
            return x[..., 1:-1, 1:-1]
        else:
            return x[..., 2:-2, 2:-2]

    def subsample_fourier(x, k):
        """Subsampling of a 2D image performed in the Fourier domain
            Subsampling in the spatial domain amounts to periodization
            in the Fourier domain, hence the formula.
            Parameters
            ----------
            x : tensor
                Input tensor with at least 5 dimensions, the last being the real
                and imaginary parts.
            k : int
                Integer such that x is subsampled by k along the spatial variables.
            Returns
            -------
            out : tensor
                Tensor such that its Fourier transform is the Fourier
                transform of a subsampled version of x, i.e. in
                F^{-1}(out)[u1, u2] = F^{-1}(x)[u1 * k, u2 * k].
        """
        batch_shape = x.shape[:-2]
        signal_shape = x.shape[-2:]
        x = x.view((-1,) + signal_shape)
        y = x.view(-1, k, signal_shape[0] // k, k, signal_shape[1] // k)

        out = y.mean((1, 3), keepdim=False)
        out = out.reshape(batch_shape + out.shape[-2:])
        return out

    def apply_filter(x, filters, cast_to_real=False):
        """ (B, C, M, N) complex and (K, M, N) real to (B, KC/CK, M, N), complex.
        Channel orders depends on whether frequencies are separated. """

        # The inline comments indicate channel shapes, first for KC order then for CK order.
        channel_order = "KC" if separate_freqs else "CK"
        x = channel_reshape(x, {"KC": (1, -1), "CK": (-1, 1)}[channel_order])  # (1, C) or (C, 1)
        filters = channel_reshape(filters[None], {"KC": (-1, 1), "CK": (1, -1)}[channel_order])  # (K, 1) or (1, K)
        y = x * filters  # (K, C) or (C, K)
        y = channel_reshape(y, (-1,))  # (KC,) or (CK,)

        if subsample:
            y = subsample_fourier(y, 2)  # (B, KC/CK, M//2, N//2)
        y = torch.fft.ifft2(y)  # (B, KC/CK, M(//2), N(//2))
        if cast_to_real:
            y = y.real  # Should be real anyway.
        y = unpad(y, subsample)  # (B, KC/CK, M(//2)-2(4), N(//2)-2(4))
        return y

    x = x_split.full_view()  # (B, C, M, N), real or complex
    if x.requires_grad:
        x.register_hook(ignore_nan_inf_gradients_hook)

    U_r = pad(x)  # (B, C, M, N)
    U_0_c = torch.fft.fft2(U_r)  # (B, C, M, N) complex

    # TODO: could merge these two in one call, but changes channel order + no cast to real.
    x_phi = apply_filter(U_0_c, phi[None], cast_to_real=not torch.is_complex(x))  # (C,), same type as x
    # Zero-sized convolutions do not work
    if psi.shape[0] > 0:
        x_psi = apply_filter(U_0_c, psi)  # (SLC,) or (CSL,), complex

    # Whether the groups are as (order, freq) or (freq, order) in lexicographical ordering, we cannot currently
    # achieve frequency separation (be it with KC or CK channel ordering) without a necessary reordering.
    # For now, we deprecate frequency separation for ease of use and maintenance, and always use CK ordering.

    # Example of failed attempt: the order of groups are reversed in the output of the scattering.
    # The C channels of x corresponds to groups (order, freq) in lexicographical order.
    # In the scattering we just ignore the frequency and then treat x as separated by orders only, without reordering.
    # Without frequency separation, we use the CK ordering which means we don't have to reorder channels.
    # With frequency separation, we use the KC ordering which does the job as well but goes to (freq, order).
    # Each order is thus convolved with all filters, in (CK) order, and we can then

    if separate_freqs:
        # x_psi is in KC order, but there's no way around the slow reordering and concatenation...

        def get_key_map(new_freqs):
            def key_map(old_key):
                old_freq, order = old_key
                if separate_freqs:
                    return [((new_freq, order), 1) for new_freq in new_freqs]
                else:
                    return [((0, order), len(new_freqs))]
            return key_map

        psi_freqs = list(range(1, psi.shape[0] + 1))

        res = dict(phi=map_group_keys(x_phi, x_split.num_channels, get_key_map(new_freqs=[0])))
        if psi.shape[0] > 0:
            res["psi"] = map_group_keys(x_psi, x_split.num_channels, get_key_map(new_freqs=psi_freqs))

    else:
        # x_psi is in CK order, no need to reorder channels: each group has its size increased by the same factor.
        res = dict(phi=SplitTensor(x_phi, groups=x_split.num_channels))
        if psi.shape[0] > 0:
            res["psi"] = SplitTensor(x_psi, groups={k: psi.shape[0] * c for k, c in x_split.num_channels.items()})

    return res


class Realifier(nn.Module):
    """ Batched module which returns C*2 real channels from C complex ones. """
    def __init__(self, input_type: TensorType):
        super().__init__()

        self.input_type = input_type
        # Because in first block, we often have real inputs even though they will be complex in the following ones.
        # Hence we treat the case where the input is real, the realifier is then the identity module.
        self.output_type = TensorType(num_channels=(2 if self.input_type.complex else 1) * self.input_type.num_channels,
                                      spatial_shape=self.input_type.spatial_shape, complex=False)

    def extra_repr(self):
        return f"input_channels={type_to_str(self.input_type)}, output_channels={type_to_str(self.output_type)}"

    def forward(self, x):
        if torch.is_complex(x):  # See comment in __init__.
            return complex_to_real_channels(x)
        else:
            return x


class Complexifier(nn.Module):
    """ Module which returns C/2 complex channels from C*2 real ones.
    Note: not batched because of non-integer channel factor. """
    def __init__(self, input_type: TensorType):
        super().__init__()

        self.input_type = input_type
        assert (not self.input_type.complex) and self.input_type.num_channels % 2 == 0
        self.output_type = TensorType(num_channels=self.input_type.num_channels // 2,
                                      spatial_shape=self.input_type.spatial_shape, complex=True)

    def extra_repr(self):
        return f"input_channels={type_to_str(self.input_type)}, output_channels={type_to_str(self.output_type)}"

    def forward(self, x):
        return real_to_complex_channels(x)


def complex_soft_thresholding(z, threshold):
    """ Returns rho_lambda(|z|) e^(i phi) = ReLU(1 - lambda/|z|) * z """
    return torch.relu(1 - threshold / (z.abs() + 1e-6)) * z


def module_collapse(z):
    """ Sets the module to 1. Returns z / |z| = e^(i phi). """
    return z / (z.abs() + 1e-6)


def module_sigmoid(z, gain, bias):
    """ Applies a sigmoid to |z|, with gain and bias to set the dead-zone, the linear zone and the saturation zane. """
    return (torch.sigmoid(gain * z.abs() + bias) / (z.abs() + 1e-6)) * z


def complex_tanh(z):
    return torch.tanh(z.abs()) / (z.abs() + 1e-6) * z


def module_power(z, gain, bias):
    """ Computes sigmoid(gain * log(|z| + bias) = t/(1 + t) with t = e^bias * |z|^gain. """
    t = (np.exp(bias) if isinstance(bias, float) else torch.exp(bias)) * z.abs() ** gain
    return (t / ((1 + t) * (z.abs() + 1e-6))) * z


class ScatNonLinearity(nn.Module):
    """ Applies a non-linearity to a real or complex input. """
    def __init__(self, input_type: SplitTensorType, non_linearity, separate_orders, gain, bias, learned_params):
        """
        :param input_type:
        :param non_linearity: can be "mod"/"abs", "relu" or "cst" (complex soft-thresholding)
        :param separate_orders: whether to separate orders, i.e., change the keys of the input after the non-linearity
        :param gain, bias: used by some non-linearities. May be None (unused) or a constant (initial value)
        :param learned_params: whether to learn params or to freeze them at their initial value
        """
        super().__init__()

        self.input_type = input_type

        self.non_linearity = non_linearity
        self.non_lin = dict(
            mod=torch.abs, abs=torch.abs, relu=torch.relu, cst=complex_soft_thresholding,
            mc=module_collapse, ms=module_sigmoid, tanh=complex_tanh, pow=module_power,
        )[non_linearity]

        def handle_param(default_value):
            if default_value is not None and learned_params:
                return nn.Parameter(torch.full((self.input_type.num_channels, 1, 1), float(default_value)))
            else:
                return float(default_value)  # None or float

        self.gain = handle_param(gain)
        self.bias = handle_param(bias)
        assert self.input_type.complex == dict(
            mod=True, abs=False, relu=False, cst=True, mc=True, ms=True, tanh=True, pow=True,
        )[non_linearity]

        self.separate_orders = separate_orders

        groups = self.handle_keys(self.input_type.groups)
        output_complex = dict(
            mod=False, abs=False, relu=False, cst=True, mc=True, ms=True, tanh=True, pow=True,
        )[non_linearity]
        self.output_type = SplitTensorType(groups=groups, spatial_shape=self.input_type.spatial_shape,
                                           complex=output_complex)

    def handle_keys(self, groups):
        def new_key(key):  # Old key (before non-linearity) to new key (after non-linearity).
            freq, order = key
            # Setting freq to 0 would require using map_group_keys, this is not done here for performance reasons.
            if self.separate_orders:
                order = order + 1
            return freq, order

        return {new_key(key): group for key, group in groups.items()}

    def extra_repr(self) -> str:
        non_lin = f"{self.non_linearity}"
        complex = f"{complex_to_str(self.input_type.complex)}2{complex_to_str(self.output_type.complex)}"
        return f"non_linearity={non_lin}, complex={complex}, separate_orders={self.separate_orders}"

    def model_info(self):
        """ Print info about biases and gains. """
        module_info = []
        for name, param in dict(bias=self.bias, gain=self.gain).items():
            if isinstance(param, nn.Parameter):
                module_info.append(f"\n  - {name.capitalize()} for {self.non_linearity}: {tensor_summary_stats(param)}")
        return module_info

    def forward(self, x: SplitTensor) -> SplitTensor:
        x_full = x.full_view()
        non_lin_kwargs = dict(  # Rebuild those each time because pointers get invalidated by DataParallel.
            mod={}, abs={}, relu={}, cst=dict(threshold=self.bias),
            mc=dict(), ms=dict(gain=self.gain, bias=self.bias), tanh=dict(), pow=dict(gain=self.gain, bias=self.bias),
        )[self.non_linearity]
        x_abs = self.non_lin(x_full, **non_lin_kwargs)
        return SplitTensor(x_abs, groups=self.handle_keys(x.num_channels))


class ScatNonLinearityAndSkip(nn.Module):
    """ z -> linear=z, non_linear=|z|. """
    def __init__(self, input_type: SplitTensorType, **non_linearity_kwargs):
        """
        :param input_type:
        """
        super().__init__()

        self.input_type = input_type

        self.non_lin = ScatNonLinearity(input_type=self.input_type, **non_linearity_kwargs)

        self.output_type = dict(linear=self.input_type, non_linear=self.non_lin.output_type)

    def __repr__(self) -> str:
        return f"SkipModulus({self.non_lin.extra_repr()})"

    def forward(self, x: SplitTensor) -> Dict[str, SplitTensor]:
        return dict(linear=x, non_linear=self.non_lin(x))
