from typing import Optional, Tuple

import torch
from torch import nn


class GCBlock(nn.Module):
    """Grouped convolution-style block with re-parameterization support.

    This block follows a multi-branch design during training and is fused
    into a single :class:`~torch.nn.Conv2d` layer for deployment. The main
    branch uses a configurable convolution kernel. Optional secondary 3x3,
    1x1 and identity branches can be enabled via the constructor. BatchNorm
    layers are fused into the resulting convolution when ``switch_to_deploy``
    is called.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        padding: Optional[int] = None,
        kernel_size: int = 3,
        use_second_3x3: bool = True,
        use_identity: bool = True,
        use_1x1: bool = True,
        use_bn: bool = True,
        deploy: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = (kernel_size // 2) if padding is None else padding
        self.use_second_3x3 = use_second_3x3
        self.use_identity = use_identity
        self.use_1x1 = use_1x1
        self.use_bn = use_bn
        self.deploy = deploy

        if deploy:
            self.reparam_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=self.padding,
                bias=True,
            )
        else:
            self.path_3x3_1 = self._conv_bn(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=self.padding,
            )

            self.path_3x3_2: Optional[nn.Sequential]
            if use_second_3x3:
                self.path_3x3_2 = self._conv_bn(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=self.padding,
                )
            else:
                self.path_3x3_2 = None

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

        out = self.path_3x3_1(x)
        if self.path_3x3_2 is not None:
            out = out + self.path_3x3_2(x)
        if self.branch_1x1 is not None:
            out = out + self.branch_1x1(x)
        if self.branch_identity is not None:
            out = out + self.branch_identity(x)
        return self.activation(out)

    def get_equivalent_kernel_bias(
        self, lambda_3x3: float = 1.0, lambda_1x1: float = 1.0, lambda_id: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kernel3x3_1, bias3x3_1 = self._fuse_conv_bn(self.path_3x3_1)

        if self.path_3x3_2 is not None:
            kernel3x3_2, bias3x3_2 = self._fuse_conv_bn(self.path_3x3_2)
        else:
            kernel3x3_2 = torch.zeros_like(kernel3x3_1)
            bias3x3_2 = torch.zeros_like(bias3x3_1)

        if self.branch_1x1 is not None:
            kernel1x1, bias1x1 = self._fuse_conv_bn(self.branch_1x1)
            kernel1x1 = self._pad_1x1_to_kxk(kernel1x1)
        else:
            kernel1x1 = torch.zeros_like(kernel3x3_1)
            bias1x1 = torch.zeros_like(bias3x3_1)

        kernelid, biasid = self._fuse_identity()

        kernel = (
            lambda_3x3 * (kernel3x3_1 + kernel3x3_2)
            + lambda_1x1 * kernel1x1
            + lambda_id * kernelid
        )
        bias = (
            lambda_3x3 * (bias3x3_1 + bias3x3_2)
            + lambda_1x1 * bias1x1
            + lambda_id * biasid
        )
        return kernel, bias

    def get_auxiliary_equivalent_kernel_bias(
        self,
        lambda_3x3: float = 1.0,
        lambda_1x1: float = 1.0,
        lambda_id: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the combined contribution of the auxiliary branches.

        This helper is used when loading fused weights back into training-time
        models while keeping the local auxiliary branches intact (variant B).
        """

        if self.path_3x3_2 is None:
            kernel3x3_2 = torch.zeros_like(self.path_3x3_1[0].weight)
            bias3x3_2 = torch.zeros_like(self.path_3x3_1[0].weight[:, 0, 0, 0])
        else:
            kernel3x3_2, bias3x3_2 = self._fuse_conv_bn(self.path_3x3_2)

        if self.branch_1x1 is None:
            kernel1x1 = torch.zeros_like(self.path_3x3_1[0].weight)
            bias1x1 = torch.zeros_like(self.path_3x3_1[0].weight[:, 0, 0, 0])
        else:
            kernel1x1, bias1x1 = self._fuse_conv_bn(self.branch_1x1)
            kernel1x1 = self._pad_1x1_to_kxk(kernel1x1)

        kernelid, biasid = self._fuse_identity()
        kernel = lambda_3x3 * kernel3x3_2 + lambda_1x1 * kernel1x1 + lambda_id * kernelid
        bias = lambda_3x3 * bias3x3_2 + lambda_1x1 * bias1x1 + lambda_id * biasid
        return kernel, bias

    def _fuse_conv_bn(self, branch: nn.Sequential) -> tuple[torch.Tensor, torch.Tensor]:
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
        """
        情况一：
            如果use_identity=None说明或者通道数不匹配导致无法建立恒等映射，这个分支就不存在
            既然不存在，它对结果的贡献就是0，即融合的时候返回一个全0的卷积核和全0的偏置
        """

        if self.branch_identity is None:
            kernel_value = torch.zeros(
                (self.out_channels, self.in_channels, self.kernel_size, self.kernel_size),
                device=self.path_3x3_1[0].weight.device,
            )
            bias_value = torch.zeros(self.out_channels, device=kernel_value.device)
            return kernel_value, bias_value
        """
        情况二：
            没有BN的纯恒等映射，为了保证输出原样图像，卷积核中心为1，周围为0即可（狄拉克函数形式的卷积核）
            bias置零
        """
        if isinstance(self.branch_identity, nn.Identity):
            identity_kernel = torch.zeros(
                (self.out_channels, self.in_channels, self.kernel_size, self.kernel_size),
                device=self.path_3x3_1[0].weight.device,
            )
            for i in range(self.out_channels):
                center = self.kernel_size // 2
                identity_kernel[i, i % self.in_channels, center, center] = 1
            bias = torch.zeros(self.out_channels, device=identity_kernel.device)
            return identity_kernel, bias
        """
        情况三：带BN的恒等映射
            首先构造基础的恒等卷积核，与情况一完全相同
            然后考虑这个BN参数的提取
            最终融合进新的卷积核即可
        """
        # BatchNorm identity branch
        input_dim = self.in_channels
        id_kernel_value = torch.zeros(
            (self.out_channels, input_dim, self.kernel_size, self.kernel_size),
            device=self.path_3x3_1[0].weight.device,
        )
        for i in range(self.out_channels):
            center = self.kernel_size // 2
            id_kernel_value[i, i % input_dim, center, center] = 1

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

    def _pad_1x1_to_kxk(self, kernel: torch.Tensor) -> torch.Tensor:
        if kernel.size(2) == 1:
            pad_total = self.kernel_size // 2
            return nn.functional.pad(kernel, [pad_total, pad_total, pad_total, pad_total])
        return kernel

    def switch_to_deploy(self) -> None:
        if self.deploy:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.reparam_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        # Clean training-time branches
        del self.path_3x3_1
        if self.path_3x3_2 is not None:
            del self.path_3x3_2
        if self.branch_1x1 is not None:
            del self.branch_1x1
        if self.branch_identity is not None:
            del self.branch_identity
        self.deploy = True


def _set_bn_to_identity(bn: nn.BatchNorm2d) -> None:
    """Reset a :class:`BatchNorm2d` layer to behave like an identity mapping."""

    bn.weight.data.fill_(1.0)
    bn.bias.data.zero_()
    bn.running_mean.zero_()
    bn.running_var.fill_(1.0)
    if hasattr(bn, "num_batches_tracked"):
        bn.num_batches_tracked.zero_()


def _zero_out_conv_branch(branch: Optional[nn.Sequential]) -> None:
    """
    Zero out a convolutional branch without changing its structure.
    将目标卷积层weight和bias置零
    服务于A思路的权重融合匹配
    """

    if branch is None:
        return
    conv = branch[0]
    conv.weight.data.zero_()
    if conv.bias is not None:
        conv.bias.data.zero_()
    if len(branch) > 1 and isinstance(branch[1], nn.BatchNorm2d):
        branch[1].weight.data.zero_()
        branch[1].bias.data.zero_()
        branch[1].running_mean.zero_()
        branch[1].running_var.fill_(1.0)
        if hasattr(branch[1], "num_batches_tracked"):
            branch[1].num_batches_tracked.zero_()


def build_fused_state_dict(
    model: nn.Module,
    lambda_3x3: float = 1.0,
    lambda_1x1: float = 1.0,
    lambda_id: float = 1.0,
) -> dict:
    """Return a deterministic mapping of fused GCBlock parameters.

    The returned dictionary maps ``"<module_name>.weight"`` and
    ``"<module_name>.bias"`` keys to tensors representing the fully fused
    convolution (including auxiliary branches). Standard parameters are kept as
    is, while GCBlock internal branches are excluded to preserve personalization.
    """

    fused_sd: dict[str, torch.Tensor] = {}
    gcblock_prefixes = {name for name, module in model.named_modules() if isinstance(module, GCBlock)}

    for key, param in model.state_dict().items():
        if any(key.startswith(prefix + ".") for prefix in gcblock_prefixes):
            continue
        fused_sd[key] = param.detach().clone()

    for name, module in model.named_modules():
        if isinstance(module, GCBlock):
            kernel, bias = module.get_equivalent_kernel_bias(
                lambda_3x3=lambda_3x3, lambda_1x1=lambda_1x1, lambda_id=lambda_id
            )
            fused_sd[f"{name}.weight"] = kernel.detach().clone()
            fused_sd[f"{name}.bias"] = bias.detach().clone()
    return fused_sd


def load_fused_weights_into_gc_model(
    model: nn.Module,
    fused_sd: dict,
    variant: str = "A",
    lambda_3x3: float = 1.0,
    lambda_1x1: float = 1.0,
    lambda_id: float = 1.0,
) -> None:
    """Load fused GCBlock weights back into a training-time model.

    Args:
        model: Model containing :class:`GCBlock` instances.
        fused_sd: Fused state dictionary produced by
            :func:`build_fused_state_dict`.
        variant: "A" zeroes auxiliary branches (1x1 and identity) each round.
            "B" preserves local auxiliary weights while adjusting the main
            3x3 branch so the combined output matches the fused weights.
    """

    variant = variant.upper()
    if variant not in {"A", "B"}:
        raise ValueError(f"Unsupported variant '{variant}'. Use 'A' or 'B'.")

    gcblock_prefixes = {name for name, module in model.named_modules() if isinstance(module, GCBlock)}
    model_sd = model.state_dict()
    for name, param in model_sd.items():
        if name in fused_sd and not any(name.startswith(prefix + ".") for prefix in gcblock_prefixes):
            param.copy_(fused_sd[name])

    for name, module in model.named_modules():
        if not isinstance(module, GCBlock):
            continue

        weight_key, bias_key = f"{name}.weight", f"{name}.bias"
        if weight_key not in fused_sd or bias_key not in fused_sd:
            raise KeyError(f"Missing fused parameters for GCBlock '{name}' in fused_sd.")

        target_kernel = fused_sd[weight_key]
        target_bias = fused_sd[bias_key]

        if module.deploy:
            module.reparam_conv.weight.data.copy_(target_kernel)
            module.reparam_conv.bias.data.copy_(target_bias)
            continue

        if variant == "A":
            _zero_out_conv_branch(module.path_3x3_2)
            _zero_out_conv_branch(module.branch_1x1)
            if isinstance(module.branch_identity, nn.BatchNorm2d):
                module.branch_identity.weight.data.zero_()
                module.branch_identity.bias.data.zero_()
                module.branch_identity.running_mean.zero_()
                module.branch_identity.running_var.fill_(1.0)
                if hasattr(module.branch_identity, "num_batches_tracked"):
                    module.branch_identity.num_batches_tracked.zero_()
            else:
                module.branch_identity = None
            residual_kernel = target_kernel
            residual_bias = target_bias
        else:
            aux_kernel, aux_bias = module.get_auxiliary_equivalent_kernel_bias(
                lambda_3x3=lambda_3x3, lambda_1x1=lambda_1x1, lambda_id=lambda_id
            )
            residual_kernel = target_kernel - aux_kernel
            residual_bias = target_bias - aux_bias

        main_conv = module.path_3x3_1[0]
        main_conv.weight.data.copy_(residual_kernel)
        if main_conv.bias is not None:
            main_conv.bias.data.copy_(residual_bias)

        if module.use_bn and len(module.path_3x3_1) > 1:
            _set_bn_to_identity(module.path_3x3_1[1])
        else:
            main_conv.bias = torch.nn.Parameter(residual_bias.clone())
