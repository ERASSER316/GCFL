from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm
import numpy as np
import math

from experiments.GCFL.Models.GCBlock import GCBlock


class CNN_1(nn.Module): # large
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        super(CNN_1, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n_kernels, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(n_kernels, 2* n_kernels, 5,padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(2* n_kernels * 7 * 7, 2000)
        self.fc2 = nn.Linear(2000, 500)
        self.fc3 = nn.Linear(500, out_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNN_2(nn.Module): # medium
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        super(CNN_2, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n_kernels, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(n_kernels, 2* n_kernels, 5,padding=2)
        self.fc1 = nn.Linear(2 * n_kernels * 7 * 7, 200)
        self.fc2 = nn.Linear(200, out_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNN_3(nn.Module): # medium
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        super(CNN_3, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n_kernels, 5,padding=2)
        self.fc1 = nn.Linear(n_kernels * 7 * 7, 200)
        self.fc2 = nn.Linear(200, out_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class GC_CNN_1(nn.Module):
    """CNN_1 variant that uses :class:`GCBlock` on the second conv stage."""

    def __init__(
        self,
        in_channels=3,
        n_kernels=16,
        out_dim=10,
        use_identity=True,
        use_1x1=True,
        use_bn=True,
        deploy=False,
    ):
        super(GC_CNN_1, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n_kernels, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = GCBlock(
            n_kernels,
            2 * n_kernels,
            padding=1,
            use_identity=use_identity,
            use_1x1=use_1x1,
            use_bn=use_bn,
            deploy=deploy,
        )
        self.fc1 = nn.Linear(2 * n_kernels * 7 * 7, 2000)
        self.fc2 = nn.Linear(2000, 500)
        self.fc3 = nn.Linear(500, out_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(self.conv2(x))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class GC_CNN_2(nn.Module):
    """CNN_2 variant that replaces the second conv with :class:`GCBlock`."""

    def __init__(
        self,
        in_channels=3,
        n_kernels=16,
        out_dim=10,
        use_identity=True,
        use_1x1=True,
        use_bn=True,
        deploy=False,
    ):
        super(GC_CNN_2, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n_kernels, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = GCBlock(
            n_kernels,
            2 * n_kernels,
            padding=1,
            use_identity=use_identity,
            use_1x1=use_1x1,
            use_bn=use_bn,
            deploy=deploy,
        )
        self.fc1 = nn.Linear(2 * n_kernels * 7 * 7, 200)
        self.fc2 = nn.Linear(200, out_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(self.conv2(x))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x




def build_model(
    model_name: str,
    use_gcblock: bool = False,
    in_channels: int = 3,
    n_kernels: int = 16,
    out_dim: int = 10,
    **gcblock_kwargs,
):
    """Instantiate baseline or GCBlock-based CNN models.

    Args:
        model_name: Name of the baseline or GC variant (e.g., ``CNN_1`` or
            ``GC_CNN_1``).
        use_gcblock: When ``True`` and a GC variant exists for the requested
            baseline, the GC version is returned.
        in_channels: Number of input channels.
        n_kernels: Base convolution width multiplier.
        out_dim: Output dimension.
        **gcblock_kwargs: Extra keyword arguments forwarded to GCBlock-enabled
            models (e.g., ``use_identity`` or ``deploy``).
    """

    baseline_builders = {
        "CNN_1": lambda: CNN_1(in_channels=in_channels, n_kernels=n_kernels, out_dim=out_dim),
        "CNN_2": lambda: CNN_2(in_channels=in_channels, n_kernels=n_kernels, out_dim=out_dim),
        "CNN_3": lambda: CNN_3(in_channels=in_channels, n_kernels=n_kernels, out_dim=out_dim),
    }

    gc_builders = {
        "GC_CNN_1": lambda: GC_CNN_1(
            in_channels=in_channels,
            n_kernels=n_kernels,
            out_dim=out_dim,
            **gcblock_kwargs,
        ),
        "GC_CNN_2": lambda: GC_CNN_2(
            in_channels=in_channels,
            n_kernels=n_kernels,
            out_dim=out_dim,
            **gcblock_kwargs,
        ),
    }

    gc_aliases = {"CNN_1": "GC_CNN_1", "CNN_2": "GC_CNN_2"}

    requested_name = model_name
    if use_gcblock and model_name in gc_aliases:
        requested_name = gc_aliases[model_name]

    if requested_name in gc_builders:
        return gc_builders[requested_name]()
    if requested_name in baseline_builders:
        return baseline_builders[requested_name]()

    known = list(baseline_builders) + list(gc_builders)
    raise ValueError(f"Unknown model '{model_name}'. Available models: {known}")