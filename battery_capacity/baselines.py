"""B3 and its N01 source-context gate, restricted to the two published arms."""
import numpy as np
import torch
from torch import nn
from .common import backbone, tensor
from .readout import PC, PH


def gamma(d):
    return .1 + .9 / (1 + torch.relu(d))


def geometry(z, pool, tau):
    w = torch.softmax(-(z[:, None] - pool[None]).square().sum(2) / 17 / tau, 1)
    center = w @ pool
    return (w * (pool[None] - center[:, None]).square().mean(2)).sum(1)


class DirectResidual(nn.Module):
    def __init__(self, method, seed, source_pool):
        super().__init__()
        if method not in ('B3', 'N01'):
            raise ValueError(method)
        self.method = method
        self.mlp = backbone(seed)
        self.mlp_h = backbone(100000 + seed)
        self.register_buffer('pool', tensor(source_pool))
        if method == 'N01':
            d = np.sum((source_pool[:, None] - source_pool[None]) ** 2, axis=2) / 17
            nonzero = d[np.triu_indices(len(source_pool), 1)]
            nonzero = nonzero[nonzero > 0]
            tau = max(float(np.median(nonzero)) if len(nonzero) else 0, 1e-6)
            self.register_buffer('tau', tensor(tau))
            w = np.exp(-d / float(self.tau) - np.max(-d / float(self.tau), axis=1, keepdims=True))
            w /= w.sum(axis=1, keepdims=True)
            means = w @ source_pool
            spread = (w * np.mean((source_pool[None] - means[:, None]) ** 2, axis=2)).sum(axis=1)
            self.register_buffer('spread_q95', tensor(np.quantile(spread, .95)))

    def forward(self, z):
        # These two arms have a zero input increment in the original graph.
        c, h = z * PC, z * PH
        c, h = c + .1 * (c * 0), h + .1 * (h * 0)
        value = .5 * (self.mlp(c).flatten() + self.mlp_h(h).flatten())
        gate = (gamma(geometry(z, self.pool, self.tau) / (self.spread_q95 + 1e-6))
                if self.method == 'N01' else z.new_ones(len(z)))
        return gate * value
