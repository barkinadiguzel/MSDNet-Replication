import torch

class CostEstimator:
    def __init__(self, exit_layers, flops_per_layer):
        self.exit_layers = exit_layers
        self.flops_per_layer = flops_per_layer

    def compute_exit_cost(self, exit_confidences, threshold=0.9):
        cumulative_cost = 0
        exit_layer = self.exit_layers[-1] 

        for idx, layer in enumerate(self.exit_layers):
            cumulative_cost += self.flops_per_layer[layer]
            if exit_confidences[idx] >= threshold:
                exit_layer = layer
                break

        return exit_layer, cumulative_cost

    def expected_cost(self, exit_probs):
        expected = 0
        cumulative = 0
        for idx, layer in enumerate(self.exit_layers):
            cumulative += self.flops_per_layer[layer]
            expected += exit_probs[idx] * cumulative
        return expected
