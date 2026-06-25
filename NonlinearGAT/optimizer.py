"""Loss helpers used by graph autoencoder variants."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Regularizer(nn.Module):
    """Small discriminator network retained for compatibility with earlier experiments."""

    def __init__(self, in_channels, hidden_dim1, hidden_dim2):
        super().__init__()
        self.dc_den1 = nn.Linear(in_channels, hidden_dim1)
        self.dc_den2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.dc_output = nn.Linear(hidden_dim2, 1)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in (self.dc_den1, self.dc_den2, self.dc_output):
            layer.bias.data.fill_(0.0)
            layer.weight.data = torch.normal(0.0, 0.001, layer.weight.shape)

    def forward(self, inputs):
        hidden = torch.sigmoid(self.dc_den1(inputs))
        hidden = torch.sigmoid(self.dc_den2(hidden))
        return self.dc_output(hidden)


def loss_function(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
    """Variational graph autoencoder reconstruction and KL loss."""
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)
    kld = -0.5 * torch.mean(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1)) / n_nodes
    return cost + kld


def loss_function1(preds, labels, norm, pos_weight):
    """Weighted binary cross-entropy reconstruction loss."""
    return norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)
