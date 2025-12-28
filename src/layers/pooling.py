import torch.nn as nn

def get_pooling(name="max", kernel_size=2, stride=2):
    if name.lower() == "max":
        return nn.MaxPool2d(kernel_size, stride)
    elif name.lower() == "avg":
        return nn.AvgPool2d(kernel_size, stride)
    else:
        raise ValueError(f"Unsupported pooling: {name}")
