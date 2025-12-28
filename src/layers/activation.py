import torch.nn as nn

def get_activation(name="relu"):
    if name.lower() == "relu":
        return nn.ReLU(inplace=True)
    elif name.lower() == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"Unsupported activation: {name}")
