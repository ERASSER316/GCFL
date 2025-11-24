from typing import Optional

import torch
from torch import nn


class GCBlock(nn.Module):
    """Grouped convolution-style block with re-parameterization support.

    This block follows a multi-branch design during training and is fused
    into a single :class:`~torch.nn.Conv2d` layer for deployment. The main
    branch uses a 3x3 convolution. Optional 1x1 and identity branches can be
    enabled via the constructor. BatchNorm layers are fused into the
    resulting convolution when ``switch_to_deploy`` is called.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        padding: int = 1,
        use_identity: bool = True,
        use_1x1: bool = True,
        use_bn: bool = True,
        deploy: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.use_identity = use_identity
        self.use_1x1 = use_1x1
        self.use_bn = use_bn
        self.deploy = deploy

        if deploy:
            self.reparam_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=padding,
                bias=True,
            )
        else:
            self.branch_3x3 = self._conv_bn(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=padding,
            )

            self.branch_1x1: Optional[nn.Sequential]
            if use_1x1:
                self.branch_1x1 = self._conv_bn(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                )
            else:
                self.branch_1x1 = None

            self.branch_identity: Optional[nn.BatchNorm2d]
            if use_identity and out_channels == in_channels and stride == 1:
                self.branch_identity = (
                    nn.BatchNorm2d(in_channels) if use_bn else nn.Identity()
                )
            else:
                self.branch_identity = None

        self.activation = nn.ReLU(inplace=True)

    def _conv_bn(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> nn.Sequential:
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=not self.use_bn,
            )
        ]
        if self.use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            return self.activation(self.reparam_conv(x))

        out = self.branch_3x3(x)
        if self.branch_1x1 is not None:
            out = out + self.branch_1x1(x)
        if self.branch_identity is not None:
            out = out + self.branch_identity(x)
        return self.activation(out)

    def get_equivalent_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        kernel3x3, bias3x3 = self._fuse_conv_bn(self.branch_3x3)

        kernel1x1 = bias1x1 = None
        if self.branch_1x1 is not None:
            kernel1x1, bias1x1 = self._fuse_conv_bn(self.branch_1x1)
            kernel1x1 = self._pad_1x1_to_3x3(kernel1x1)
        else:
            kernel1x1 = torch.zeros_like(kernel3x3)
            bias1x1 = torch.zeros_like(bias3x3)

        kernelid, biasid = self._fuse_identity()

        kernel = kernel3x3 + kernel1x1 + kernelid
        bias = bias3x3 + bias1x1 + biasid
        return kernel, bias

    def _fuse_conv_bn(
        self, branch: nn.Sequential
    ) -> tuple[torch.Tensor, torch.Tensor]:
        conv = branch[0]
        bn = branch[1] if self.use_bn and len(branch) > 1 else None

        if bn is None:
            weight = conv.weight
            bias = conv.bias if conv.bias is not None else torch.zeros_like(weight[:, 0, 0, 0])
            return weight, bias

        std = torch.sqrt(bn.running_var + bn.eps)
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        fused_weight = conv.weight * t
        fused_bias = bn.bias - bn.running_mean * bn.weight / std
        if conv.bias is not None:
            fused_bias += conv.bias * bn.weight / std
        return fused_weight, fused_bias

    def _fuse_identity(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.branch_identity is None:
            kernel_value = torch.zeros(
                (self.out_channels, self.in_channels, 3, 3), device=self.branch_3x3[0].weight.device
            )
            bias_value = torch.zeros(self.out_channels, device=kernel_value.device)
            return kernel_value, bias_value

        if isinstance(self.branch_identity, nn.Identity):
            identity_kernel = torch.zeros(
                (self.out_channels, self.in_channels, 3, 3), device=self.branch_3x3[0].weight.device
            )
            for i in range(self.out_channels):
                identity_kernel[i, i % self.in_channels, 1, 1] = 1
            bias = torch.zeros(self.out_channels, device=identity_kernel.device)
            return identity_kernel, bias

        # BatchNorm identity branch
        input_dim = self.in_channels
        id_kernel_value = torch.zeros(
            (self.out_channels, input_dim, 3, 3), device=self.branch_3x3[0].weight.device
        )
        for i in range(self.out_channels):
            id_kernel_value[i, i % input_dim, 1, 1] = 1

        running_mean = self.branch_identity.running_mean
        running_var = self.branch_identity.running_var
        gamma = self.branch_identity.weight
        beta = self.branch_identity.bias
        eps = self.branch_identity.eps
        std = torch.sqrt(running_var + eps)
        t = (gamma / std).reshape(-1, 1, 1, 1)
        fused_kernel = id_kernel_value * t
        fused_bias = beta - running_mean * gamma / std
        return fused_kernel, fused_bias

    @staticmethod
    def _pad_1x1_to_3x3(kernel: torch.Tensor) -> torch.Tensor:
        if kernel.size(2) == 1:
            return nn.functional.pad(kernel, [1, 1, 1, 1])
        return kernel

    def switch_to_deploy(self) -> None:
        if self.deploy:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.reparam_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=3,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        # Clean training-time branches
        del self.branch_3x3
        if self.branch_1x1 is not None:
            del self.branch_1x1
        if self.branch_identity is not None:
            del self.branch_identity
        self.deploy = True
