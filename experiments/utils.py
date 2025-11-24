import argparse
import logging
import random

import numpy as np
import torch


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def set_logger():
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )


def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def state_dict_parameter_count(state_dict):
    return sum(tensor.numel() for tensor in state_dict.values())


def state_dict_num_bytes(state_dict):
    return sum(tensor.numel() * tensor.element_size() for tensor in state_dict.values())


def format_num_bytes(num_bytes: float) -> str:
    suffixes = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    while num_bytes >= 1024 and idx < len(suffixes) - 1:
        num_bytes /= 1024
        idx += 1
    return f"{num_bytes:.2f} {suffixes[idx]}"


def get_device(no_cuda=False, gpus='0'):
    return torch.device(f"cuda:{gpus}" if torch.cuda.is_available() and not no_cuda else "cpu")
