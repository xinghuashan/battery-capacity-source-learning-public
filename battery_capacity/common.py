"""Shared numerical operations used by the paper models."""
import hashlib
import numpy as np
import torch
from torch import nn

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

def tensor(value):
    return torch.as_tensor(value, dtype=torch.float64)


def state_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def backbone(seed):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = nn.Sequential(nn.Linear(17, 16), nn.SiLU(), nn.Linear(16, 16),
                              nn.SiLU(), nn.Linear(16, 1))
        nn.init.zeros_(model[-1].weight)
        nn.init.zeros_(model[-1].bias)
    return model


def balanced(values, cells):
    return torch.stack([values[cells == c].mean(dim=0) for c in torch.unique(cells)]).mean(dim=0)

def weights(groups):
    names, counts = np.unique(groups, return_counts=True)
    lookup = dict(zip(names, counts))
    return np.array([len(groups) / (len(names) * lookup[g]) for g in groups])
