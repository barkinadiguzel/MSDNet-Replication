import torch

def dense_concat(features_list):
    return torch.cat(features_list, dim=1)
