import torch
import torch.nn as nn
import numpy as np
from mmengine.model import BaseModule
from torch import Tensor
from typing import Optional, Tuple, Union
from mmseg.utils import OptConfigType
from mmcv.cnn import ConvModule, build_norm_layer, build_activation_layer


class Block1x1(BaseModule):
    """
    GCBlock的1×1-1×1分支模块（训练时双1×1卷积+BN，推理时融合为单1×1卷积）
    功能：通过双1×1卷积实现通道维度变换，同时支持训练/推理模式切换（重参数化）
    """
    
    def __init__(self,
                in_channels: int,
                out_channels: int,
                stride: Union[int, Tuple[int]] = 1,
                padding: Union[int, Tuple[int]] = 0,
                bias: bool = True,
                norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
                deploy: bool = False):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            stride: 卷积步长（默认1，1×1卷积步长不影响空间尺寸）
            padding: 卷积填充（默认0，1×1卷积无需填充）
            bias: 是否使用偏置（训练时False，推理时True，因BN已融合偏置）
            norm_cfg: 归一化层配置（默认BN，训练时生效）
            deploy: 是否为推理模式（True=单1×1卷积，False=双1×1卷积+BN）
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.deploy = deploy
        
        # 推理模式：直接使用单1×1卷积（已融合训练时双分支参数）
        if self.deploy:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=1,
                stride=stride, padding=padding, bias=True
            )
        # 训练模式：双1×1卷积+BN（ConvModule=Conv2d+BN+激活，此处act_cfg=None无激活）
        else:
            self.conv1 = ConvModule(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                stride=stride, padding=padding, bias=bias, norm_cfg=norm_cfg, act_cfg=None
            )
            self.conv2 = ConvModule(
                in_channels=out_channels, out_channels=out_channels, kernel_size=1,
                stride=1, padding=padding, bias=bias, norm_cfg=norm_cfg, act_cfg=None
            )
    
    def forward(self, x):
        """前向传播：推理时单卷积，训练时双卷积串联"""
        if self.deploy:
            return self.conv(x)
        else:
            x = self.conv1(x)
            x = self.conv2(x)
            return x
    
    def _fuse_bn_tensor(self, conv: ConvModule):
        """
        融合ConvModule中的卷积层与BN层（核心重参数化操作）
        公式：融合后权重 = 卷积权重 * (BN_gamma / sqrt(BN_var+eps))
             融合后偏置 = BN_beta + (卷积偏置 - BN_mean) * BN_gamma / sqrt(BN_var+eps)
        """
        # 提取卷积层与BN层参数
        kernel = conv.conv.weight  # 卷积权重 (out, in, 1, 1)
        bias = conv.conv.bias if conv.conv.bias is not None else 0  # 卷积偏置
        running_mean = conv.bn.running_mean  # BN移动均值
        running_var = conv.bn.running_var    # BN移动方差
        gamma = conv.bn.weight               # BN缩放系数
        beta = conv.bn.bias                  # BN偏置
        eps = conv.bn.eps                    # BN防止除零的小值
        
        # 计算融合参数
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)  # 适配卷积权重维度的缩放因子
        fused_kernel = kernel * t
        fused_bias = beta + (bias - running_mean) * gamma / std if self.bias else beta - running_mean * gamma / std
        return fused_kernel, fused_bias
    
    def switch_to_deploy(self):
        """
        训练模式切换为推理模式：
        1. 融合conv1与conv2的卷积+BN参数
        2. 计算双1×1卷积串联的等效单1×1卷积参数
        3. 替换为单卷积层，删除训练时分支
        """
        # 融合conv1与conv2的参数
        kernel1, bias1 = self._fuse_bn_tensor(self.conv1)
        kernel2, bias2 = self._fuse_bn_tensor(self.conv2)
        
        # 计算双1×1卷积串联的等效权重（矩阵乘法）与偏置（累加+加权）
        # kernel2 (O, M, 1,1) * kernel1 (M, I, 1,1) = 等效权重 (O, I, 1,1)
        fused_kernel = torch.einsum('oi,icjk->ocjk', kernel2.squeeze(3).squeeze(2), kernel1)
        # 等效偏置 = bias2 + sum(bias1 * kernel2的通道权重)
        fused_bias = bias2 + (bias1.view(1, -1, 1, 1) * kernel2).sum(3).sum(2).sum(1)
        
        # 构建推理用单卷积层
        self.conv = nn.Conv2d(
            self.in_channels, self.out_channels, kernel_size=1,
            stride=self.stride, padding=self.padding, bias=True
        )
        self.conv.weight.data = fused_kernel
        self.conv.bias.data = fused_bias
        
        # 删除训练时的双分支
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        self.deploy = True


class Block3x3(BaseModule):
    """
    GCBlock的3×3-1×1分支模块（训练时3×3卷积+1×1卷积+BN，推理时融合为单3×3卷积）
    功能：捕捉局部空间特征（3×3）+ 通道维度压缩（1×1），支持重参数化
    """
    
    def __init__(self,
                in_channels: int,
                out_channels: int,
                stride: Union[int, Tuple[int]] = 1,
                padding: Union[int, Tuple[int]] = 0,
                bias: bool = True,
                norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
                deploy: bool = False):
        """
        Args:
            padding: 3×3卷积的填充（默认0，训练时需外部适配same padding）
            其他参数同Block1x1
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.deploy = deploy
        
        # 推理模式：单3×3卷积（融合3×3+1×1参数）
        if self.deploy:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=3,
                stride=stride, padding=padding, bias=True
            )
        # 训练模式：3×3卷积（抓空间）+ 1×1卷积（压通道）+ BN
        else:
            self.conv1 = ConvModule(
                in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                stride=stride, padding=padding, bias=bias, norm_cfg=norm_cfg, act_cfg=None
            )
            self.conv2 = ConvModule(
                in_channels=out_channels, out_channels=out_channels, kernel_size=1,
                stride=1, padding=0, bias=bias, norm_cfg=norm_cfg, act_cfg=None
            )
    
    def forward(self, x):
        """前向传播：推理时单3×3卷积，训练时3×3→1×1串联"""
        if self.deploy:
            return self.conv(x)
        else:
            x = self.conv1(x)
            x = self.conv2(x)
            return x
    
    def _fuse_bn_tensor(self, conv: ConvModule):
        """同Block1x1，融合卷积与BN参数（适配3×3卷积权重维度）"""
        kernel = conv.conv.weight  # (out, in, 3, 3)
        bias = conv.conv.bias if conv.conv.bias is not None else 0
        running_mean = conv.bn.running_mean
        running_var = conv.bn.running_var
        gamma = conv.bn.weight
        beta = conv.bn.bias
        eps = conv.bn.eps
        
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        fused_kernel = kernel * t
        fused_bias = beta + (bias - running_mean) * gamma / std if self.bias else beta - running_mean * gamma / std
        return fused_kernel, fused_bias
    
    def switch_to_deploy(self):
        """
        训练→推理模式切换：
        融合3×3+1×1卷积的参数，生成等效单3×3卷积
        """
        kernel1, bias1 = self._fuse_bn_tensor(self.conv1)  # 3×3卷积融合参数
        kernel2, bias2 = self._fuse_bn_tensor(self.conv2)  # 1×1卷积融合参数
        
        # 计算3×3→1×1串联的等效权重（1×1卷积权重适配3×3维度后与3×3权重矩阵乘法）
        # kernel2 (O, M, 1,1) → 展平为(O,M)，kernel1 (M, I, 3,3) → 等效为(O, I, 3,3)
        fused_kernel = torch.einsum('oi,icjk->ocjk', kernel2.squeeze(3).squeeze(2), kernel1)
        # 等效偏置 = bias2 + sum(bias1 * kernel2的通道权重)
        fused_bias = bias2 + (bias1.view(1, -1, 1, 1) * kernel2).sum(3).sum(2).sum(1)
        
        # 构建推理用单3×3卷积
        self.conv = nn.Conv2d(
            self.in_channels, self.out_channels, kernel_size=3,
            stride=self.stride, padding=self.padding, bias=True
        )
        self.conv.weight.data = fused_kernel
        self.conv.bias.data = fused_bias
        
        # 删除训练分支
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        self.deploy = True


class GCBlock(nn.Module):
    """
    核心模块：Group Convolution Block（GCBlock）
    设计思想：训练时多分支并行（2个3×3-1×1分支+1个1×1-1×1分支+残差分支），
    推理时重参数化为单3×3卷积，平衡训练表达能力与推理效率
    """
    
    def __init__(self,
                in_channels: int,
                out_channels: int,
                kernel_size: Union[int, Tuple[int]] = 3,
                stride: Union[int, Tuple[int]] = 1,
                padding: Union[int, Tuple[int]] = 1,
                padding_mode: Optional[str] = 'zeros',
                norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
                act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                act: bool = True,
                deploy: bool = False):
        """
        Args:
            kernel_size: 卷积核大小（强制为3，因模块设计基于3×3卷积）
            padding: 3×3卷积的same padding（强制为1，确保输入输出尺寸一致）
            act: 是否使用激活函数（默认True，输出前ReLU）
            其他参数同Block1x1/Block3x3
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.deploy = deploy
        
        # 强制校验：GCBlock仅支持3×3 kernel与1 padding（确保same padding和模块设计一致性）
        assert kernel_size == 3, "GCBlock only supports kernel_size=3"
        assert padding == 1, "GCBlock only supports padding=1 for same output size"
        
        padding_11 = padding - kernel_size // 2  # 1×1分支的padding（3//2=1，故padding_11=0）
        
        # 激活函数（默认ReLU，无激活时用恒等映射）
        self.relu = build_activation_layer(act_cfg) if act else nn.Identity()
        
        # 推理模式：单3×3卷积（融合所有训练分支参数）
        if deploy:
            self.reparam_3x3 = nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, bias=True, padding_mode=padding_mode
            )
        # 训练模式：4个并行分支（2×3×3-1×1 + 1×1×1-1×1 + 残差）
        else:
            # 残差分支：仅当输入输出通道一致且步长为1时存在（BN层，无卷积，保留原始特征）
            if (out_channels == in_channels) and (stride == 1):
                self.path_residual = build_norm_layer(norm_cfg, num_features=in_channels)[1]
            else:
                self.path_residual = None  # 通道/步长不匹配时无残差
            
            # 分支1：3×3-1×1（抓局部空间特征1）
            self.path_3x3_1 = Block3x3(
                in_channels=in_channels, out_channels=out_channels,
                stride=stride, padding=padding, bias=False, norm_cfg=norm_cfg
            )
            # 分支2：3×3-1×1（抓局部空间特征2，与分支1参数独立，增强表达）
            self.path_3x3_2 = Block3x3(
                in_channels=in_channels, out_channels=out_channels,
                stride=stride, padding=padding, bias=False, norm_cfg=norm_cfg
            )
            # 分支3：1×1-1×1（抓通道维度特征，补充空间分支的不足）
            self.path_1x1 = Block1x1(
                in_channels=in_channels, out_channels=out_channels,
                stride=stride, padding=padding_11, bias=False, norm_cfg=norm_cfg
            )
    
    def forward(self, inputs: Tensor) -> Tensor:
        """
        前向传播：
        - 推理时：单3×3卷积 → 激活
        - 训练时：4分支求和 → 激活
        """
        # 推理模式
        if hasattr(self, 'reparam_3x3'):
            return self.relu(self.reparam_3x3(inputs))
        
        # 训练模式：计算各分支输出
        # 残差分支：有则用BN处理输入，无则输出0
        id_out = self.path_residual(inputs) if self.path_residual is not None else 0
        # 3个卷积分支输出求和 + 残差分支
        total_out = self.path_3x3_1(inputs) + self.path_3x3_2(inputs) + self.path_1x1(inputs) + id_out
        # 激活后输出
        return self.relu(total_out)
    
    def get_equivalent_kernel_bias(self):
        """
        计算所有训练分支的等效权重与偏置（重参数化核心）
        步骤：
        1. 所有子分支切换到推理模式，获取融合后的参数
        2. 1×1分支权重填充为3×3维度（便于与其他分支求和）
        3. 残差分支转换为等效3×3恒等卷积参数
        4. 所有分支参数求和，得到单3×3卷积的等效参数
        """
        # 1. 3×3-1×1分支1：获取融合后的3×3参数
        self.path_3x3_1.switch_to_deploy()
        kernel3x3_1, bias3x3_1 = self.path_3x3_1.conv.weight.data, self.path_3x3_1.conv.bias.data
        
        # 2. 3×3-1×1分支2：同上
        self.path_3x3_2.switch_to_deploy()
        kernel3x3_2, bias3x3_2 = self.path_3x3_2.conv.weight.data, self.path_3x3_2.conv.bias.data
        
        # 3. 1×1-1×1分支：获取融合后的1×1参数，填充为3×3
        self.path_1x1.switch_to_deploy()
        kernel1x1, bias1x1 = self.path_1x1.conv.weight.data, self.path_1x1.conv.bias.data
        kernel1x1_padded = self._pad_1x1_to_3x3_tensor(kernel1x1)
        
        # 4. 残差分支：转换为等效3×3恒等卷积参数
        kernelid, biasid = self._fuse_bn_tensor(self.path_residual)
        
        # 所有分支参数求和（权重+偏置）
        equivalent_kernel = kernel3x3_1 + kernel3x3_2 + kernel1x1_padded + kernelid
        equivalent_bias = bias3x3_1 + bias3x3_2 + bias1x1 + biasid
        
        return equivalent_kernel, equivalent_bias
    
    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """将1×1卷积权重填充为3×3维度（上下左右各补1个0，中心保留原权重）"""
        if kernel1x1 is None:
            return 0
        else:
            # padding格式：[左,右,上,下]，填充后尺寸从( O,I,1,1 )→( O,I,3,3 )
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])
    
    def _fuse_bn_tensor(self, conv: Union[nn.BatchNorm2d, None]):
        """
        将残差分支的BN层转换为等效3×3恒等卷积参数
        原理：BN层对输入x的操作 → x*gamma/sqrt(var+eps) + (beta - mean*gamma/sqrt(var+eps))
        等效为恒等卷积（中心权重1，其余0）与BN参数融合后的3×3卷积
        """
        if conv is None:
            return 0, 0  # 无残差分支时返回0
        
        # 若为ConvModule（扩展场景），提取对应参数
        if isinstance(conv, ConvModule):
            kernel = conv.conv.weight
            running_mean = conv.bn.running_mean
            running_var = conv.bn.running_var
            gamma = conv.bn.weight
            beta = conv.bn.bias
            eps = conv.bn.eps
        # 若为BN层（残差分支默认场景），构造恒等卷积权重
        else:
            assert isinstance(conv,
                            (nn.SyncBatchNorm, nn.BatchNorm2d)), "Only BN layers are supported for residual path"
            # 构造恒等卷积权重：(out, in, 3, 3)，仅中心位置为1（其他为0）
            if not hasattr(self, 'id_tensor'):
                kernel_value = np.zeros((self.out_channels, self.in_channels, 3, 3), dtype=np.float32)
                for i in range(self.out_channels):
                    kernel_value[i, i % self.in_channels, 1, 1] = 1  # 中心位置权重=1
                self.id_tensor = torch.from_numpy(kernel_value).to(conv.weight.device)
            kernel = self.id_tensor
            running_mean = conv.running_mean
            running_var = conv.running_var
            gamma = conv.weight
            beta = conv.bias
            eps = conv.eps
        
        # 融合BN参数到恒等卷积权重与偏置
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        fused_kernel = kernel * t
        fused_bias = beta - running_mean * gamma / std  # 残差分支无卷积偏置，故简化公式
        
        return fused_kernel, fused_bias
    
    def switch_to_deploy(self):
        """
        训练模式切换为推理模式：
        1. 计算所有分支的等效参数
        2. 构建单3×3卷积层，赋值等效参数
        3. 删除训练时的所有分支，释放内存
        """
        if hasattr(self, 'reparam_3x3'):
            return  # 已为推理模式，无需重复操作
        
        # 获取等效参数
        kernel, bias = self.get_equivalent_kernel_bias()
        
        # 构建推理用单3×3卷积
        self.reparam_3x3 = nn.Conv2d(
            in_channels=self.in_channels, out_channels=self.out_channels,
            kernel_size=self.kernel_size, stride=self.stride,
            padding=self.padding, bias=True
        )
        self.reparam_3x3.weight.data = kernel
        self.reparam_3x3.bias.data = bias
        
        # 冻结参数（推理时无需梯度）
        for para in self.parameters():
            para.detach_()
        
        # 删除训练时的分支与临时变量
        self.__delattr__('path_3x3_1')
        self.__delattr__('path_3x3_2')
        self.__delattr__('path_1x1')
        if hasattr(self, 'path_residual'):
            self.__delattr__('path_residual')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        
        self.deploy = True


def test_gcblock():
    """测试GCBlock的基本功能"""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(1, 64, 32, 32).to(device)
    
    # 创建GCBlock实例
    model = GCBlock(64, 64)
    model.to(device)
    
    # 训练模式前向传播
    y_train = model(x)
    print("训练模式测试:")
    print("输入特征维度：", x.shape)
    print("输出特征维度：", y_train.shape)
    
    # 切换到推理模式
    model.switch_to_deploy()   
    y_deploy = model(x)
    print("\n推理模式测试:")
    print("输出特征维度：", y_deploy.shape)
    
    # 检查输出是否一致（允许微小误差）
    diff = torch.abs(y_train - y_deploy).max().item()
    print(f"\n训练与推理模式输出最大差异: {diff:.6f}")
    
    print("\n微信公众号：十小大的底层视觉工坊")
    print("知乎、CSDN：十小大")


if __name__ == "__main__":
    test_gcblock()
