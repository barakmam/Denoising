"""
MobileNetV3 Architecture
Using sources:
TODO
"""

import torch
import collections.abc as container_abcs
from collections import OrderedDict
from functools import partial
import math
from typing import List, Tuple, Optional
from itertools import repeat
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from dropblock import DropBlock2D
from dropblock import LinearScheduler as DropBlockScheduled

# ======================================================================================================================

def sigmoid(x, inplace: bool = False):
    return x.sigmoid_() if inplace else x.sigmoid()


"""
Layer/Module Helpers
Hacked together by Ross Wightman
"""

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


tup_single = _ntuple(1)
tup_pair = _ntuple(2)
tup_triple = _ntuple(3)
tup_quadruple = _ntuple(4)

""" Padding Helpers

Hacked together by Ross Wightman
"""


# import torch.nn.functional as F


# Calculate symmetric padding for a convolution
def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


# Calculate asymmetric TensorFlow-like 'SAME' padding for a convolution
def get_same_padding(x: int, k: int, s: int, d: int):
    return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)


# Can SAME padding for given args be done statically?
def is_static_pad(kernel_size: int, stride: int = 1, dilation: int = 1, **_):
    return stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0


# Dynamically pad input x with 'SAME' padding for conv with specified args
def pad_same(x, k: List[int], s: List[int], d: List[int] = (1, 1), value: float = 0):
    ih, iw = x.size()[-2:]
    pad_h, pad_w = get_same_padding(ih, k[0], s[0], d[0]), get_same_padding(iw, k[1], s[1], d[1])
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], value=value)
    return x


def get_padding_value(padding, kernel_size, **kwargs) -> Tuple[Tuple, bool]:
    dynamic = False
    if isinstance(padding, str):
        # for any string padding, the padding will be calculated for you, one of three ways
        padding = padding.lower()
        if padding == 'same':
            # TF compatible 'SAME' padding, has a performance and GPU memory allocation impact
            if is_static_pad(kernel_size, **kwargs):
                # static case, no extra overhead
                padding = get_padding(kernel_size, **kwargs)
            else:
                # dynamic 'SAME' padding, has runtime/GPU memory overhead
                padding = 0
                dynamic = True
        elif padding == 'valid':
            # 'VALID' padding, same as padding=0
            padding = 0
        else:
            # Default to PyTorch style 'same'-ish symmetric padding
            padding = get_padding(kernel_size, **kwargs)
    return padding, dynamic


""" Conv2d w/ Same Padding
Hacked together by Ross Wightman
"""


def conv2d_same(
        x, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0), dilation: Tuple[int, int] = (1, 1), groups: int = 1):
    x = pad_same(x, weight.shape[-2:], stride, dilation)
    return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)


class Conv2dSame(nn.Conv2d):
    """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dSame, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        return conv2d_same(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def create_conv2d_pad(in_chs, out_chs, kernel_size, **kwargs):
    padding = kwargs.pop('padding', '')
    kwargs.setdefault('bias', False)
    padding, is_dynamic = get_padding_value(padding, kernel_size, **kwargs)
    if is_dynamic:
        return Conv2dSame(in_chs, out_chs, kernel_size, **kwargs)
    else:
        return nn.Conv2d(in_chs, out_chs, kernel_size, padding=padding, **kwargs)


""" PyTorch Mixed Convolution

Paper: MixConv: Mixed Depthwise Convolutional Kernels (https://arxiv.org/abs/1907.09595)

Hacked together by Ross Wightman
"""


def _split_channels(num_chan, num_groups):
    split = [num_chan // num_groups for _ in range(num_groups)]
    split[0] += num_chan - sum(split)
    return split


class MixedConv2d(nn.ModuleDict):
    """ Mixed Grouped Convolution

    Based on MDConv and GroupedConv in MixNet impl:
      https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/custom_layers.py
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding='', dilation=1, depthwise=False, **kwargs):
        super(MixedConv2d, self).__init__()

        kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size]
        num_groups = len(kernel_size)
        in_splits = _split_channels(in_channels, num_groups)
        out_splits = _split_channels(out_channels, num_groups)
        self.in_channels = sum(in_splits)
        self.out_channels = sum(out_splits)
        for idx, (k, in_ch, out_ch) in enumerate(zip(kernel_size, in_splits, out_splits)):
            conv_groups = out_ch if depthwise else 1
            # use add_module to keep key space clean
            self.add_module(
                str(idx),
                create_conv2d_pad(
                    in_ch, out_ch, k, stride=stride,
                    padding=padding, dilation=dilation, groups=conv_groups, **kwargs)
            )
        self.splits = in_splits

    def forward(self, x):
        x_split = torch.split(x, self.splits, 1)
        x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
        x = torch.cat(x_out, 1)
        return x


""" PyTorch Conditionally Parameterized Convolution (CondConv)

Paper: CondConv: Conditionally Parameterized Convolutions for Efficient Inference
(https://arxiv.org/abs/1904.04971)

Hacked together by Ross Wightman
"""


def get_condconv_initializer(initializer, num_experts, expert_shape):
    def condconv_initializer(weight):
        """CondConv initializer function."""
        num_params = np.prod(expert_shape)
        if (len(weight.shape) != 2 or weight.shape[0] != num_experts or
                weight.shape[1] != num_params):
            raise (ValueError(
                'CondConv variables must have shape [num_experts, num_params]'))
        for i in range(num_experts):
            initializer(weight[i].view(expert_shape))

    return condconv_initializer


class CondConv2d(nn.Module):
    """ Conditionally Parameterized Convolution
    Inspired by: https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/condconv/condconv_layers.py

    Grouped convolution hackery for parallel execution of the per-sample kernel filters inspired by this discussion:
    https://github.com/pytorch/pytorch/issues/17983
    """
    __constants__ = ['in_channels', 'out_channels', 'dynamic_padding']

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding='', dilation=1, groups=1, bias=False, num_experts=4):
        super(CondConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = tup_pair(kernel_size)
        self.stride = tup_pair(stride)
        padding_val, is_padding_dynamic = get_padding_value(
            padding, kernel_size, stride=stride, dilation=dilation)
        self.dynamic_padding = is_padding_dynamic  # if in forward to work with torchscript
        self.padding = tup_pair(padding_val)
        self.dilation = tup_pair(dilation)
        self.groups = groups
        self.num_experts = num_experts

        self.weight_shape = (self.out_channels, self.in_channels // self.groups) + self.kernel_size
        weight_num_param = 1
        for wd in self.weight_shape:
            weight_num_param *= wd
        self.weight = torch.nn.Parameter(torch.Tensor(self.num_experts, weight_num_param))

        if bias:
            self.bias_shape = (self.out_channels,)
            self.bias = torch.nn.Parameter(torch.Tensor(self.num_experts, self.out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        init_weight = get_condconv_initializer(
            partial(nn.init.kaiming_uniform_, a=math.sqrt(5)), self.num_experts, self.weight_shape)
        init_weight(self.weight)
        if self.bias is not None:
            fan_in = np.prod(self.weight_shape[1:])
            bound = 1 / math.sqrt(fan_in)
            init_bias = get_condconv_initializer(
                partial(nn.init.uniform_, a=-bound, b=bound), self.num_experts, self.bias_shape)
            init_bias(self.bias)

    def forward(self, x, routing_weights):
        B, C, H, W = x.shape
        weight = torch.matmul(routing_weights, self.weight)
        new_weight_shape = (B * self.out_channels, self.in_channels // self.groups) + self.kernel_size
        weight = weight.view(new_weight_shape)
        bias = None
        if self.bias is not None:
            bias = torch.matmul(routing_weights, self.bias)
            bias = bias.view(B * self.out_channels)
        # move batch elements with channels so each batch element can be efficiently convolved with separate kernel
        x = x.view(1, B * C, H, W)
        if self.dynamic_padding:
            out = conv2d_same(
                x, weight, bias, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups * B)
        else:
            out = F.conv2d(
                x, weight, bias, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups * B)
        out = out.permute([1, 0, 2, 3]).view(B, self.out_channels, out.shape[-2], out.shape[-1])

        # Literal port (from TF definition)
        # x = torch.split(x, 1, 0)
        # weight = torch.split(weight, 1, 0)
        # if self.bias is not None:
        #     bias = torch.matmul(routing_weights, self.bias)
        #     bias = torch.split(bias, 1, 0)
        # else:
        #     bias = [None] * B
        # out = []
        # for xi, wi, bi in zip(x, weight, bias):
        #     wi = wi.view(*self.weight_shape)
        #     if bi is not None:
        #         bi = bi.view(*self.bias_shape)
        #     out.append(self.conv_fn(
        #         xi, wi, bi, stride=self.stride, padding=self.padding,
        #         dilation=self.dilation, groups=self.groups))
        # out = torch.cat(out, 0)
        return out


""" Create Conv2d Factory Method

Hacked together by Ross Wightman
"""


def create_conv2d(in_channels, out_channels, kernel_size, **kwargs):
    """ Select a 2d convolution implementation based on arguments
    Creates and returns one of torch.nn.Conv2d, Conv2dSame, MixedConv2d, or CondConv2d.

    Used extensively by EfficientNet, MobileNetv3 and related networks.
    """
    if isinstance(kernel_size, list):
        assert 'num_experts' not in kwargs  # MixNet + CondConv combo not supported currently
        assert 'groups' not in kwargs  # MixedConv groups are defined by kernel list
        # We're going to use only lists for defining the MixedConv2d kernel groups,
        # ints, tuples, other iterables will continue to pass to normal conv and specify h, w.
        m = MixedConv2d(in_channels, out_channels, kernel_size, **kwargs)
    else:
        depthwise = kwargs.pop('depthwise', False)
        groups = out_channels if depthwise else kwargs.pop('groups', 1)
        if 'num_experts' in kwargs and kwargs['num_experts'] > 0:
            m = CondConv2d(in_channels, out_channels, kernel_size, groups=groups, **kwargs)
        else:
            m = create_conv2d_pad(in_channels, out_channels, kernel_size, groups=groups, **kwargs)
    return m


class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, dilation=1, pad_type='', act_layer=nn.ReLU,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(ConvBnAct, self).__init__()
        norm_kwargs = norm_kwargs or {}
        self.conv = create_conv2d(in_chs, out_chs, kernel_size, stride=stride, dilation=dilation, padding=pad_type)
        self.bn1 = norm_layer(out_chs, **norm_kwargs)
        self.act1 = act_layer(inplace=True)

    def feature_info(self, location):
        if location == 'expansion' or location == 'depthwise':
            # no expansion or depthwise this block, use act after conv
            info = dict(module='act1', hook_type='forward', num_chs=self.conv.out_channels)
        else:  # location == 'bottleneck'
            info = dict(module='', hook_type='', num_chs=self.conv.out_channels)
        return info

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    random_tensor = keep_prob + torch.rand((x.size()[0], 1, 1, 1), dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


def swish(x):
    return x * x.sigmoid()


def hard_sigmoid(x, inplace=False):
    return F.relu6(x + 3, inplace) / 6


def hard_swish(x, inplace=False):
    return x * hard_sigmoid(x, inplace)


class HardSigmoid(nn.Module):
    def __init__(self, inplace=False):
        super(HardSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_sigmoid(x, inplace=self.inplace)


class HardSwish(nn.Module):
    def __init__(self, inplace=False):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_swish(x, inplace=self.inplace)


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# https://github.com/jonnedtc/Squeeze-Excitation-PyTorch/blob/master/networks.py
class SqEx(nn.Module):

    def __init__(self, n_features, reduction=4):
        super(SqEx, self).__init__()

        if n_features % reduction != 0:
            raise ValueError('n_features must be divisible by reduction (default = 4)')

        self.linear1 = nn.Linear(n_features, n_features // reduction, bias=True)
        self.nonlin1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(n_features // reduction, n_features, bias=True)
        self.nonlin2 = HardSigmoid(inplace=True)

    def forward(self, x):
        y = F.avg_pool2d(x, kernel_size=x.size()[2:4])
        y = y.permute(0, 2, 3, 1)
        y = self.nonlin1(self.linear1(y))
        y = self.nonlin2(self.linear2(y))
        y = y.permute(0, 3, 1, 2)
        y = x * y
        return y


class LinearBottleneck(nn.Module):
    def __init__(self, inplanes, outplanes, expplanes, k=3, stride=1, drop_prob=0, num_steps=3e5, start_step=0,
                 activation=nn.ReLU, act_params={"inplace": True}, SE=False):
        super(LinearBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, expplanes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expplanes)
        self.db1 = DropBlockScheduled(DropBlock2D(drop_prob=0, block_size=7), start_value=0.,
                                      stop_value=drop_prob, nr_steps=num_steps)
        self.act1 = activation(**act_params)  # first does have act according to MobileNetV2

        self.conv2 = nn.Conv2d(expplanes, expplanes, kernel_size=k, stride=stride, padding=k // 2, bias=False,
                               groups=expplanes)
        self.bn2 = nn.BatchNorm2d(expplanes)
        self.db2 = DropBlockScheduled(DropBlock2D(drop_prob=drop_prob, block_size=7), start_value=0.,
                                      stop_value=drop_prob, nr_steps=num_steps)
        self.act2 = activation(**act_params)

        self.se = SqEx(expplanes) if SE else lambda x: x

        self.conv3 = nn.Conv2d(expplanes, outplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.db3 = DropBlockScheduled(DropBlock2D(drop_prob=drop_prob, block_size=7), start_value=0.,
                                      stop_value=drop_prob, nr_steps=num_steps)
        # self.act3 = activation(**act_params)  # works worse

        self.stride = stride
        self.expplanes = expplanes
        self.inplanes = inplanes
        self.outplanes = outplanes

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.db1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.db2(out)
        out = self.act2(out)

        out = self.se(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.db3(out)
        # out = self.act3(out)

        if self.stride == 1 and self.inplanes == self.outplanes:  # TODO: or add 1x1?
            out += residual  # No inplace if there is in-place activation before

        return out


class LastBlockLarge(nn.Module):
    def __init__(self, inplanes, num_classes, expplanes1, expplanes2):
        super(LastBlockLarge, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, expplanes1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expplanes1)
        self.act1 = HardSwish(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.conv2 = nn.Conv2d(expplanes1, expplanes2, kernel_size=1, stride=1)
        self.act2 = HardSwish(inplace=True)

        self.dropout = nn.Dropout(p=0.2, inplace=True)
        self.fc = nn.Linear(expplanes2, num_classes)

        self.expplanes1 = expplanes1
        self.expplanes2 = expplanes2
        self.inplanes = inplanes
        self.num_classes = num_classes

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.avgpool(out)

        out = self.conv2(out)
        out = self.act2(out)

        # flatten for input to fully-connected layer
        out = out.view(out.size(0), -1)
        out = self.fc(self.dropout(out))

        return out


class LastBlockSmall(nn.Module):
    def __init__(self, inplanes, num_classes, expplanes1, expplanes2):
        super(LastBlockSmall, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, expplanes1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expplanes1)
        self.act1 = HardSwish(inplace=True)

        self.se = SqEx(expplanes1)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.conv2 = nn.Conv2d(expplanes1, expplanes2, kernel_size=1, stride=1, bias=False)
        self.act2 = HardSwish(inplace=True)

        self.dropout = nn.Dropout(p=0.2, inplace=True)
        self.fc = nn.Linear(expplanes2, num_classes)

        self.expplanes1 = expplanes1
        self.expplanes2 = expplanes2
        self.inplanes = inplanes
        self.num_classes = num_classes

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.se(out)
        out = self.avgpool(out)

        out = self.conv2(out)
        out = self.act2(out)

        # flatten for input to fully-connected layer
        out = out.view(out.size(0), -1)
        out = self.fc(self.dropout(out))

        return out


class DepthwiseSeparableConv(nn.Module):
    """ DepthwiseSeparable block
    Used for DS convs in MobileNet-V1 and in the place of IR blocks that have no expansion
    (factor of 1.0). This is an alternative to having a IR with an optional first pw conv.
    """

    def __init__(self, in_chs, out_chs, dw_kernel_size=3,
                 stride=1, dilation=1, pad_type='', act_layer=nn.ReLU, noskip=False,
                 pw_kernel_size=1, pw_act=False, se_ratio=0., se_kwargs=None,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None, drop_path_rate=0.):
        super(DepthwiseSeparableConv, self).__init__()
        norm_kwargs = norm_kwargs or {}
        has_se = se_ratio is not None and se_ratio > 0.
        self.has_residual = (stride == 1 and in_chs == out_chs) and not noskip
        self.has_pw_act = pw_act  # activation after point-wise conv
        self.drop_path_rate = drop_path_rate

        self.conv_dw = create_conv2d(
            in_chs, in_chs, dw_kernel_size, stride=stride, dilation=dilation, padding=pad_type, depthwise=True)
        self.bn1 = norm_layer(in_chs, **norm_kwargs)
        self.act1 = act_layer(inplace=True)

        self.se = None

        self.conv_pw = create_conv2d(in_chs, out_chs, pw_kernel_size, padding=pad_type)
        self.bn2 = norm_layer(out_chs, **norm_kwargs)
        self.act2 = act_layer(inplace=True) if self.has_pw_act else nn.Identity()

    def feature_info(self, location):
        if location == 'expansion':
            # no expansion in this block, use depthwise, before SE
            info = dict(module='act1', hook_type='forward', num_chs=self.conv_pw.in_channels)
        elif location == 'depthwise':  # after SE
            info = dict(module='conv_pw', hook_type='forward_pre', num_chs=self.conv_pw.in_channels)
        else:  # location == 'bottleneck'
            info = dict(module='', hook_type='', num_chs=self.conv_pw.out_channels)
        return info

    def forward(self, x):
        residual = x

        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.act1(x)

        if self.se is not None:
            x = self.se(x)

        x = self.conv_pw(x)
        x = self.bn2(x)
        x = self.act2(x)

        if self.has_residual:
            if self.drop_path_rate > 0.:
                x = drop_path(x, self.drop_path_rate, self.training)
            x += residual
        return x


class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, dilation=1, pad_type='', act_layer=nn.ReLU,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(ConvBnAct, self).__init__()
        norm_kwargs = norm_kwargs or {}
        self.conv = create_conv2d(in_chs, out_chs, kernel_size, stride=stride, dilation=dilation, padding=pad_type)
        self.bn1 = norm_layer(out_chs, **norm_kwargs)
        self.act1 = act_layer(inplace=True)

    def feature_info(self, location):
        if location == 'expansion' or location == 'depthwise':
            # no expansion or depthwise this block, use act after conv
            info = dict(module='act1', hook_type='forward', num_chs=self.conv.out_channels)
        else:  # location == 'bottleneck'
            info = dict(module='', hook_type='', num_chs=self.conv.out_channels)
        return info

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x

# ======================================================================================================================
""" Our modifications here! """

class MobileNetV3(nn.Module):
    """
    MobileNetV3 implementation.
    """

    def __init__(self, input_shape, in_channels=3, scaling_depth=3, start_step=0, num_steps=int(3e5), device='cpu'):
        super(MobileNetV3, self).__init__()

        self.scaling_depth = scaling_depth
        self.start_step = start_step
        self.num_steps = num_steps

        assert((input_shape >= 2 ** scaling_depth) and (scaling_depth >= 1), "Invalid scaling_depth")

        self.ds = DepthwiseSeparableConv(in_chs=in_channels, out_chs=16, dw_kernel_size=3,
                                         stride=1, dilation=1, pad_type='', act_layer=nn.ReLU, noskip=False,
                                         pw_kernel_size=1, pw_act=False, se_ratio=0., se_kwargs=None,
                                         norm_layer=nn.BatchNorm2d, norm_kwargs=None, drop_path_rate=0.).to(device)

        # setting of bottlenecks blocks
        self.bottlenecks_setting = [
            # in, exp, out, s, k,         dp,    se,      act
            [16, 16 * 3, 24, 2, 5, 0, False, nn.ReLU],  # -> size/2
            [24, 24 * 3, 24, 1, 5, 0, True, nn.ReLU]  # -> size/2
                ]
        if scaling_depth >= 2:
            self.bottlenecks_setting = self.bottlenecks_setting + [
                [24, 24 * 3, 40, 2, 5, 0, False, nn.ReLU],  # -> size/4
                [40, 40 * 6, 40, 1, 5, 0, True, nn.ReLU]  # -> size/4
            ]

        if scaling_depth >= 3:
            self.bottlenecks_setting = self.bottlenecks_setting + [
                [40, 40 * 6, 80, 2, 5, 0, True, nn.ReLU],  # -> size/8
                [80, 80 * 6, 80, 1, 5, 0, True, nn.ReLU],  # -> size/8
                [80, 80 * 6, 112, 1, 5, 0, True, nn.ReLU]  # -> size/8
            ]

        if scaling_depth >= 4:  # 4 is maximal depth
            self.bottlenecks_setting = self.bottlenecks_setting + [
                [112, 112 * 6, 112, 1, 5, 0, True, nn.ReLU],  # -> size/8
                [112, 112 * 6, 192, 2, 5, 0, True, nn.ReLU],  # -> size/16
                [192, 192 * 6, 192, 1, 5, 0, True, nn.ReLU]  # -> size/16
            ]

        self.bottleneck_out_channles = self.bottlenecks_setting[-1][2]

        for l in self.bottlenecks_setting:
            l[0] = _make_divisible(l[0], 8)
            l[1] = _make_divisible(l[1], 8)
            l[2] = _make_divisible(l[2], 8)

        self.bottlenecks = self._make_bottlenecks().to(device)

        self.last_layer = ConvBnAct(in_chs=self.bottleneck_out_channles, out_chs=960, kernel_size=1,
                                    stride=1, dilation=1, pad_type='', act_layer=nn.ReLU,
                                    norm_layer=nn.BatchNorm2d, norm_kwargs=None).to(device)

        self.num_features = self._get_conv_output((in_channels, input_shape, input_shape))

        decoder = [nn.ConvTranspose2d(in_channels=960, out_channels=192, kernel_size=4, stride=2, padding=1)]  # -> size*2
        if scaling_depth == 2:
            decoder.append(nn.ConvTranspose2d(in_channels=192, out_channels=in_channels, kernel_size=4, stride=2, padding=1))  # -> size*4
        elif scaling_depth == 3:
            decoder.append(nn.ConvTranspose2d(in_channels=192, out_channels=80, kernel_size=4, stride=2, padding=1))  # -> size*4
            decoder.append(nn.ConvTranspose2d(in_channels=80, out_channels=in_channels, kernel_size=4, stride=2, padding=1))  # -> size*8
        else:  # 4 is maximal depth
            decoder.append(nn.ConvTranspose2d(in_channels=192, out_channels=80, kernel_size=4, stride=2, padding=1))  # -> size*4
            decoder.append(nn.ConvTranspose2d(in_channels=80, out_channels=24, kernel_size=4, stride=2, padding=1))  # -> size*8
            decoder.append(nn.ConvTranspose2d(in_channels=24, out_channels=in_channels, kernel_size=4, stride=2, padding=1))  # -> size*16
        self.decode = nn.Sequential(decoder)

        # belongs to classifier - we don't need this:
        # self.classifier = nn.Sequential(
        #     nn.Dropout(),
        #     nn.Linear(self.num_features, num_classes)
        # )

    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape)).cuda()

        input = self.ds(input)
        input = self.bottlenecks(input)
        input = self.last_layer(input)
        # output_feat = self.features(input)  # belongs to classifier - we don't need this
        n_size = input.data.view(batch_size, -1).size(1)
        return n_size

    def _make_bottlenecks(self):
        modules = OrderedDict()
        stage_name = "Bottleneck"

        # add LinearBottleneck
        for i, setup in enumerate(self.bottlenecks_setting):
            name = stage_name + "_{}".format(i)
            module = LinearBottleneck(setup[0], setup[2], setup[1], k=setup[4], stride=setup[3], drop_prob=setup[5],
                                      num_steps=self.num_steps, start_step=self.start_step, activation=setup[7],
                                      act_params={"inplace": True}, SE=setup[6])
            modules[name] = module

        return nn.Sequential(modules)

    def forward(self, x):

        x = self.ds(x)
        x = self.bottlenecks(x)
        x = self.last_layer(x)
        x = self.decode(x)
        # belongs to classifier - we don't need this:
        # x = x.view(x.size(0), -1)
        # x = self.classifier(x)
        return x

def mobilenetv3(input_size=224, in_channels=3, scaling_depth=3, weights_dict=None, device='cpu'):
    model = MobileNetV3(input_shape=input_size, in_channels=in_channels, scaling_depth=scaling_depth, device=device)
    if weights_dict is not None:
        model.load_state_dict(weights_dict)
    return model
