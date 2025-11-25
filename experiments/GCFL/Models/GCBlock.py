from typing import Optional, Tuple

from typing import Optional, Tuple

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

    def get_auxiliary_equivalent_kernel_bias(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the combined contribution of the auxiliary branches.

        This helper is used when loading fused weights back into training-time
        models while keeping the local auxiliary branches intact (variant B).
        """

        if self.branch_1x1 is None:
            kernel1x1 = torch.zeros_like(self.branch_3x3[0].weight)
            bias1x1 = torch.zeros_like(self.branch_3x3[0].weight[:, 0, 0, 0])
        else:
            kernel1x1, bias1x1 = self._fuse_conv_bn(self.branch_1x1)
            kernel1x1 = self._pad_1x1_to_3x3(kernel1x1)

        kernelid, biasid = self._fuse_identity()
        kernel = kernel1x1 + kernelid
        bias = bias1x1 + biasid
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
        """
        情况一：
            如果use_identity=None说明或者通道数不匹配导致无法建立恒等映射，这个分支就不存在
            既然不存在，它对结果的贡献就是0，即融合的时候返回一个全0的卷积核和全0的偏置
        """

        if self.branch_identity is None:
            kernel_value = torch.zeros(
                (self.out_channels, self.in_channels, 3, 3), device=self.branch_3x3[0].weight.device
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
                (self.out_channels, self.in_channels, 3, 3), device=self.branch_3x3[0].weight.device
            )
            for i in range(self.out_channels):
                identity_kernel[i, i % self.in_channels, 1, 1] = 1
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


def build_fused_state_dict(model: nn.Module) -> dict:   #找到模型中GCBlock的部分，进行重参数化后，将等效的权重返回，修剪权重的关键
    """Return a deterministic mapping of fused GCBlock parameters.

    The returned dictionary maps ``"<module_name>.weight"`` and
    ``"<module_name>.bias"`` keys to tensors representing the fully fused
    3x3 convolution (including 1x1 and identity branches).
    """

    fused_sd: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, GCBlock):
            kernel, bias = module.get_equivalent_kernel_bias()
            fused_sd[f"{name}.weight"] = kernel.detach().clone()
            fused_sd[f"{name}.bias"] = bias.detach().clone()
    return fused_sd


def load_fused_weights_into_gc_model(
    model: nn.Module, fused_sd: dict, variant: str = "A"
) -> None:
    """Load fused GCBlock weights back into a training-time model.

    Args:
        model: Model containing :class:`GCBlock` instances.
        fused_sd: Fused state dictionary produced by
            :func:`build_fused_state_dict`.
        variant: "A" zeroes auxiliary branches (1x1 and identity) each round.
            "B" preserves local auxiliary weights while adjusting the main
            3x3 branch so the combined output matches the fused weights.

        "A"：每轮都把 1×1 分支和 identity 分支清零，全局融合权重全部写到 3×3 分支里；

        "B"：保留本地的辅助分支（1×1、identity），只调 3×3 这条主分支，让“三者加起来的等效卷积”刚好等于 fused 权重。
    """

    variant = variant.upper()
    if variant not in {"A", "B"}:
        raise ValueError(f"Unsupported variant '{variant}'. Use 'A' or 'B'.")

    for name, module in model.named_modules():  #找到GCBlock部分
        if not isinstance(module, GCBlock):
            continue    #不是GCBlock就跳过

        weight_key, bias_key = f"{name}.weight", f"{name}.bias"
        if weight_key not in fused_sd or bias_key not in fused_sd:
            raise KeyError(f"Missing fused parameters for GCBlock '{name}' in fused_sd.")

        target_kernel = fused_sd[weight_key]    #取融合后的等效卷积核（3×3 Conv 的权重）
        target_bias = fused_sd[bias_key]    #取融合后的偏置

        # Deploy-mode blocks can load the fused weights directly.
        if module.deploy:
            module.reparam_conv.weight.data.copy_(target_kernel)
            module.reparam_conv.bias.data.copy_(target_bias)
            continue

        if variant == "A":
            # Zero auxiliary branches so the 3x3 branch carries the fused weights.
            """
            核心思想：
                让 “总输出 = 3×3 分支 + 1×1 分支 + identity 分支 = target 等效卷积”
                现在我希望：1×1 分支 + identity 分支 = 0，那么 3×3 分支 = target 等效卷积。
                下次训练时候1*1分支和identity分支从零开始训练
            """
            #处理1*1分支，将其置零
            _zero_out_conv_branch(module.branch_1x1)
            #处理identity分支
            if isinstance(module.branch_identity, nn.BatchNorm2d):
                module.branch_identity.weight.data.zero_()
                module.branch_identity.bias.data.zero_()
                module.branch_identity.running_mean.zero_()
                module.branch_identity.running_var.fill_(1.0)
                if hasattr(module.branch_identity, "num_batches_tracked"):
                    module.branch_identity.num_batches_tracked.zero_()
            else:
                module.branch_identity = None
            #由于其他分支被归零，所以“总效果全靠等效 3×3 分支”，
            residual_kernel = target_kernel
            residual_bias = target_bias
        else:  # variant "B"
            """
            核心思想：
                先把当前 1×1 + identity 分支的等效卷积算出来（aux_kernel, aux_bias），
                然后：
                    3x3_residual = target_kernel - aux_kernel
                    3x3_bias_residual = target_bias - aux_bias
                这样“3×3 分支 + 辅助分支 = target 效果”。
            """
            aux_kernel, aux_bias = module.get_auxiliary_equivalent_kernel_bias()
            residual_kernel = target_kernel - aux_kernel
            residual_bias = target_bias - aux_bias

        # Write residual into the 3x3 branch with identity BatchNorm.
        main_conv = module.branch_3x3[0]
        main_conv.weight.data.copy_(residual_kernel)
        if main_conv.bias is not None:
            main_conv.bias.data.copy_(residual_bias)

        if module.use_bn and len(module.branch_3x3) > 1:
            _set_bn_to_identity(module.branch_3x3[1])
        else:
            # If BN is disabled and no conv bias exists, register bias via BN-like tensor.
            # This path is unlikely but keeps logic consistent.
            main_conv.bias = torch.nn.Parameter(residual_bias.clone())
