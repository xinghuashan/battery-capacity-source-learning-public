"""Two-branch features, support normalization, and differentiable Ridge head."""
import torch
from torch import nn
from .common import backbone, balanced, state_hash, tensor

PC = tensor([1]*13+[0]*4)


PH = tensor([0]*12+[1]*5)

class Features(nn.Module):
    def __init__(self,seed):
        super().__init__()
        self.c=nn.Sequential(*list(backbone(seed).children())[:4])
        self.h=nn.Sequential(*list(backbone(100000+seed).children())[:4])
        self.seed=seed
        self.initial_hash=state_hash(self.state_dict())

    def forward(self,z):
        half=z.new_full((len(z),1),.5)
        return torch.cat((self.c(z*PC)/2,half,self.h(z*PH)/2,half),1)


class FrozenHeads(nn.Module):
    def __init__(self,seed):
        super().__init__()
        self.features=Features(seed)
        for parameter in self.features.parameters():parameter.requires_grad_(False)
        self.readout=nn.Parameter(torch.zeros(34))

    def forward(self,z):return self.features(z)@self.readout


def normalization(model,episode):
    raw=model(episode['u'])
    s2=balanced(raw.square().sum(1)/34,episode['u_ids'])
    assert torch.isfinite(s2) and float(s2.detach())>=1/68-1e-15
    return s2


def solve_readout(model,episode,record=None,identity=None):
    s2=normalization(model,episode)
    phi=model(episode['z_l'])/s2.sqrt()
    w,t=episode['weights'],episode['t']
    gram=phi.T@(w[:,None]*phi)
    matrix=gram+torch.eye(34,dtype=phi.dtype)
    rhs=phi.T@(w*t)
    if record:record('readout_solve_started',identity=identity)
    factor=torch.linalg.cholesky(matrix)
    coefficients=torch.cholesky_solve(rhs[:,None],factor)[:,0]
    residual=(matrix@coefficients-rhs).norm()/rhs.norm().clamp_min(1e-30)
    assert torch.isfinite(coefficients).all() and float(residual.detach())<1e-10
    if record:record('readout_solve_completed',identity=identity,relative_normal_equation_residual=float(residual.detach()))
    return coefficients,s2,gram,residual


def episode_prediction(model,episode,record=None,identity=None):
    coefficients,s2,gram,residual=solve_readout(model,episode,record,identity)
    prediction=model(episode['query_z'])/s2.sqrt()@coefficients
    return prediction,dict(coefficients=coefficients,s2=s2,gram=gram,solve_residual=residual)


def objective(model, episodes, arm, record=None, identity=None):
    risks, baselines, states = ([], [], [])
    for episode in episodes:
        prediction, state = episode_prediction(model, episode, record, dict(identity or {}, episode=episode['id']))
        error = episode['query_t'] - prediction
        risk = error.square().mean()
        weighted = (error.square() * episode['query_weights']).sum()
        risks.append(weighted if arm in ('M04', 'M06') else risk)
        baselines.append(episode['query_t'].square().mean())
        states.append(state)
    risk = torch.stack(risks)
    base = torch.stack(baselines)
    main = risk.sum() if arm in ('M04', 'M06') else risk.mean()
    penalty = main.new_zeros(())
    active = []
    return (main + 0.1 * penalty, dict(main=main, penalty=penalty, risks=risk, baseline_risks=base, active=active, states=states))
