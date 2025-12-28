import torch
import torch.nn as nn
from backbone.multi_scale_blocks import MultiScaleBlock
from classifiers.early_exit import EarlyExit

class MSDNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=100, num_scales=3, depth=4, channels=32, exit_layers=None, exit_thresholds=None):
        super().__init__()
        self.num_scales = num_scales
        self.depth = depth
        self.channels = channels

        self.init_blocks = nn.ModuleList([nn.Conv2d(in_channels, channels, 3, padding=1) for _ in range(num_scales)])

        self.blocks = nn.ModuleList([
            MultiScaleBlock(
                in_channels_list=[channels]*num_scales,
                out_channels_list=[channels]*num_scales,
                stride_list=[1]*num_scales
            ) for _ in range(depth)
        ])

        if exit_layers is None:
            exit_layers = list(range(depth))
        self.exit_layers = exit_layers
        self.classifiers = nn.ModuleList([EarlyExit(channels, num_classes) for _ in exit_layers])

        if exit_thresholds is None:
            exit_thresholds = [0.9]*len(exit_layers)
        self.exit_thresholds = exit_thresholds

    def forward(self, x):
        features = [block(x) for block in self.init_blocks]
        batch_size = x.size(0)
        outputs = torch.zeros(batch_size, self.classifiers[0].out_features, device=x.device)
        exited = torch.zeros(batch_size, dtype=torch.bool, device=x.device)

        for i, block in enumerate(self.blocks):
            features = block(features)

            if i in self.exit_layers:
                idx = self.exit_layers.index(i)
                logits = self.classifiers[idx](features[-1])
                probs = torch.softmax(logits, dim=1)
                max_conf, _ = torch.max(probs, dim=1)

                for j in range(batch_size):
                    if not exited[j] and max_conf[j] >= self.exit_thresholds[idx]:
                        outputs[j] = logits[j]
                        exited[j] = True

        final_logits = self.classifiers[-1](features[-1])
        for j in range(batch_size):
            if not exited[j]:
                outputs[j] = final_logits[j]

        return outputs
