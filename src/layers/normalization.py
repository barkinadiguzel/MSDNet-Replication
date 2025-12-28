import torch.nn as nn

def get_norm(name, channels):
    if name.lower() == "batchnorm":
        return nn.BatchNorm2d(channels)
    elif name.lower() == "layernorm":
        return nn.LayerNorm([channels, 1, 1])
    else:
        raise ValueError(f"Unsupported normalization: {name}")
