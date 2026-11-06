"""Thirteen paper estimators, with all selection restricted to source cells."""
import numpy as np
import torch
from sklearn.linear_model import Ridge
from .baselines import DirectResidual
from .common import balanced, tensor
from .episodes import approved, direct_source, fit_anchor, fit_outer_anchor, make_episode, pseudo_episodes
from .kernels import choose_grid, kernel, krr_solve, positive_length_scale
from .readout import Features, FrozenHeads, objective, solve_readout

METHODS = ('Ridge', 'Ridge-CV', 'KRR', 'KRR-wide', 'B3', 'N01', 'M01',
           'M02', 'M03', 'M04', 'M05', 'M06', 'Raw-CV')
DETERMINISTIC = {'Ridge', 'Ridge-CV', 'KRR', 'KRR-wide', 'M01'}
GRID = (1e-4, 1e-3, 1e-2, .1, 1., 10., 100., 1000.)
ETAS = (.5, 1., 2.)


class Predictor:
    """Frozen source predictor; prediction accepts input rows only."""
    def __init__(self, method, scaler, anchor, model=None, coefficients=None,
                 scale=1., selection=None):
        self.method, self.scaler, self.anchor = method, scaler, anchor
        self.model, self.coefficients, self.scale = model, coefficients, scale
        self.selection = selection

    def predict(self, x):
        z = self.scaler.transform(np.asarray(x, dtype=float))
        base = self.anchor.predict(z)
        with torch.no_grad():
            if self.method in ('Ridge', 'Ridge-CV'):
                delta = np.zeros(len(z))
            elif self.method == 'M01':
                delta = self.model.predict(z)
            elif self.method in ('KRR', 'KRR-wide'):
                m = self.model
                delta = kernel(z, m['z'], m['eta'], m['ell']) @ m['beta']
            elif self.method in ('B3', 'N01', 'M02'):
                delta = self.model(tensor(z)).numpy()
            else:
                delta = (self.model(tensor(z)) / self.scale @ tensor(self.coefficients)).numpy()
        return base + delta / 100


def raw_solve(model, episode, lam):
    phi = model(episode['z_l'])
    gram = phi.T @ (episode['weights'][:, None] * phi)
    matrix = gram + lam * torch.eye(34)
    rhs = phi.T @ (episode['weights'] * episode['t'])
    factor = torch.linalg.cholesky(matrix)
    beta = torch.cholesky_solve(rhs[:, None], factor)[:, 0]
    residual = (matrix @ beta - rhs).norm() / rhs.norm().clamp_min(1e-30)
    assert torch.isfinite(beta).all() and float(residual.detach()) < 1e-8
    return beta


def train_raw(episodes, outer, lam, seed):
    model = Features(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    for _ in range(300):
        optimizer.zero_grad(set_to_none=True)
        risks = []
        for episode in episodes:
            beta = raw_solve(model, episode, lam)
            risks.append((episode['query_t'] - model(episode['query_z']) @ beta).square().mean())
        value = torch.stack(risks).mean()
        assert torch.isfinite(value)
        value.backward()
        optimizer.step()
    with torch.no_grad():
        beta = raw_solve(model, outer, lam)
    return model, beta.detach().numpy()


def select_raw(source):
    groups = np.unique(source['ids'])
    nested = []
    for held in groups:
        support = source['ids'] != held
        scaler, anchor = fit_anchor(source, support)
        validation = make_episode(source, support, ~support, scaler, anchor, 'validation', held)
        outer = make_episode(source, support, support, scaler, anchor, 'nested_outer', -1)
        episodes = []
        for query_cell in groups[groups != held]:
            query = source['ids'] == query_cell
            train = support & ~query
            inner_scaler, inner_anchor = fit_anchor(source, train)
            episodes.append(make_episode(source, train, query, inner_scaler, inner_anchor,
                                         f'nested_query_{query_cell}', query_cell))
        nested.append((episodes, outer, validation))
    candidates = []
    for lam in GRID:
        scores = []
        for episodes, outer, validation in nested:
            model, beta = train_raw(episodes, outer, lam, seed=0)
            with torch.no_grad():
                prediction = (model(validation['query_z']) @ tensor(beta)).numpy()
            scores.append(float(np.sqrt(np.mean((validation['query_t'].numpy() - prediction) ** 2))))
        candidates.append(dict(lambda_value=lam, per_cell_rmse_pp=scores,
                               source_macro_query_rmse_pp=float(np.mean(scores))))
    selected = min(candidates, key=lambda r: (r['source_macro_query_rmse_pp'], -r['lambda_value']))
    return dict(candidates=candidates, selected=selected, selection_seed=0)


def fit_method(bundle, method, seed=0, selection_cache=None):
    if method not in METHODS or seed not in (0, 1, 2, 3, 4):
        raise ValueError((method, seed))
    if method in DETERMINISTIC and seed != 0:
        raise ValueError('Deterministic estimators use a single fit.')
    cache = {} if selection_cache is None else selection_cache
    source, outer = bundle['source'], bundle['outer']
    scaler, anchor = bundle['scaler'], bundle['anchor']
    if method == 'Ridge':
        return Predictor(method, scaler, anchor)
    if method == 'Ridge-CV':
        candidates = []
        y = approved(source)
        for alpha in GRID:
            scores = []
            for held in np.unique(source['ids']):
                query = source['ids'] == held
                s, r = fit_anchor(source, ~query, alpha)
                labeled = query & source['chosen']
                prediction = r.predict(s.transform(source['x'][labeled]))
                scores.append(float(np.sqrt(np.mean((100 * (prediction - y[labeled])) ** 2))))
            candidates.append(dict(alpha=alpha, per_cell_rmse_pp=scores,
                                   source_macro_query_rmse_pp=float(np.mean(scores))))
        choice = min(candidates, key=lambda r: (r['source_macro_query_rmse_pp'], -r['alpha']))
        scaler, anchor = fit_anchor(source, np.ones(len(y), dtype=bool), choice['alpha'])
        return Predictor(method, scaler, anchor, selection=dict(candidates=candidates, selected=choice))
    if method in ('KRR', 'KRR-wide'):
        candidates = []
        for eta in ETAS:
            for lam in ((.1, 1., 10.) if method == 'KRR' else GRID):
                scores = []
                for episode in bundle['real']:
                    ell = positive_length_scale(episode['u'].numpy())
                    fitted = krr_solve(episode['z_l'].numpy(), episode['t'].numpy(),
                                       episode['weights'].numpy(), ell, eta, lam)
                    prediction = kernel(episode['query_z'].numpy(), fitted['z'], eta, ell) @ fitted['beta']
                    scores.append(float(np.sqrt(np.mean((episode['query_t'].numpy() - prediction) ** 2))))
                candidates.append(dict(eta=eta, lambda_value=lam, source_query_rmse_pp=scores,
                                       source_macro_query_rmse_pp=float(np.mean(scores))))
        choice = choose_grid(candidates)
        fitted = krr_solve(outer['z_l'].numpy(), outer['t'].numpy(), outer['weights'].numpy(),
                           positive_length_scale(outer['u'].numpy()), choice['eta'], choice['lambda_value'])
        return Predictor(method, scaler, anchor, model=fitted,
                         selection=dict(candidates=candidates, selected=choice))
    if method == 'M01':
        residual = Ridge(alpha=1).fit(outer['z_l'].numpy(), outer['t'].numpy(),
                                      sample_weight=outer['weights'].numpy())
        return Predictor(method, scaler, anchor, model=residual)
    if method in ('B3', 'N01'):
        scaler, anchor, outer = direct_source(source)
        model = DirectResidual(method, seed, outer['u'].numpy())
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
        ids = torch.as_tensor(source['ids'][source['chosen']])
        for _ in range(300):
            optimizer.zero_grad(set_to_none=True)
            value = balanced((outer['t'] - model(outer['z_l'])).square(), ids)
            assert torch.isfinite(value)
            value.backward()
            optimizer.step()
        return Predictor(method, scaler, anchor, model=model)
    if method == 'M02':
        scaler, anchor = fit_outer_anchor(source)
        full = np.ones(len(source['x']), dtype=bool)
        outer = make_episode(source, full, full, scaler, anchor, 'outer', -1)
        model = FrozenHeads(seed)
        optimizer = torch.optim.AdamW([model.readout], lr=.001, weight_decay=.001)
        for _ in range(300):
            optimizer.zero_grad(set_to_none=True)
            value = (outer['weights'] * (outer['t'] - model(outer['z_l'])).square()).sum() / len(outer['t'])
            assert torch.isfinite(value)
            value.backward()
            optimizer.step()
        return Predictor(method, scaler, anchor, model=model)
    if method == 'Raw-CV':
        if 'raw' not in cache:
            cache['raw'] = select_raw(source)
        selection = cache['raw']
        model, beta = train_raw(bundle['real'], outer, selection['selected']['lambda_value'], seed)
        return Predictor(method, scaler, anchor, model=model, coefficients=beta, selection=selection)
    model = Features(seed)
    if method != 'M03':
        if method == 'M04':
            episodes = [outer]
        elif method == 'M05':
            episodes = bundle['real']
        else:
            if 'pseudo' not in cache:
                cache['pseudo'] = pseudo_episodes(source)
            episodes = cache['pseudo'][0]
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
        for _ in range(300):
            optimizer.zero_grad(set_to_none=True)
            value, _ = objective(model, episodes, method)
            assert torch.isfinite(value)
            value.backward()
            optimizer.step()
    beta, s2, _, _ = solve_readout(model, outer)
    return Predictor(method, scaler, anchor, model=model, coefficients=beta.detach().numpy(),
                     scale=np.sqrt(float(s2.detach())))
