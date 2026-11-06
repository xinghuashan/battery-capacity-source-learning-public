"""Support-only scaling, weighted anchors, and physical/pseudo-cell episodes."""
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from .common import tensor, weights
from .grouping import pseudo_assign


def approved(source):
    values = np.full(len(source['x']), np.nan)
    values[source['chosen']] = source['approved_y']
    return values


def fit_anchor(source, support, alpha=1.):
    labeled = support & source['chosen']
    assert labeled.any() and np.isfinite(approved(source)[labeled]).all()
    scaler = StandardScaler().fit(source['x'][support])
    ridge = Ridge(alpha=alpha).fit(scaler.transform(source['x'][labeled]),
                                 approved(source)[labeled],
                                 sample_weight=weights(source['ids'][labeled]))
    return scaler, ridge


def fit_outer_anchor(source):
    """Preserve the original column-wise outer-anchor path.

    The original feature assembly is Fortran-contiguous. Fitting the scaler
    before any row indexing preserves its floating-point reduction order.
    """
    x = np.asfortranarray(source['x'])
    chosen = source['chosen']
    scaler = StandardScaler().fit(x)
    ridge = Ridge(alpha=1.).fit(scaler.transform(x[chosen]), source['approved_y'],
                                sample_weight=weights(source['ids'][chosen]))
    return scaler, ridge


def direct_source(source):
    """B3/N01 transform the whole source pool before selecting label rows."""
    scaler, ridge = fit_outer_anchor(source)
    x = np.asfortranarray(source['x'])
    chosen = source['chosen']
    z = scaler.transform(x)
    payload = dict(u=tensor(z), z_l=tensor(z)[torch.as_tensor(chosen)],
                   t=tensor(100*(source['approved_y']-ridge.predict(z[chosen]))),
                   weights=tensor(weights(source['ids'][chosen])))
    return scaler, ridge, payload


def make_episode(source, support, query, scaler, ridge, identity, held_group):
    labels = approved(source)
    train_l = np.flatnonzero(support & source['chosen'])
    query_l = np.flatnonzero(query & source['chosen'])
    assert len(train_l) and len(query_l)
    z_u = scaler.transform(source['x'][support])
    z_l = scaler.transform(source['x'][train_l])
    z_query = scaler.transform(source['x'][query_l])
    global_l = source['ids'][source['chosen']]
    m = len(np.unique(global_l))
    qw = np.array([1 / (m * int((global_l == source['ids'][i]).sum())) for i in query_l])
    return dict(id=identity, held_group=int(held_group),
                u=tensor(z_u), u_ids=torch.as_tensor(source['ids'][support]),
                z_l=tensor(z_l), t=tensor(100 * (labels[train_l] - ridge.predict(z_l))),
                weights=tensor(weights(source['ids'][train_l])),
                query_z=tensor(z_query),
                query_t=tensor(100 * (labels[query_l] - ridge.predict(z_query))),
                query_weights=tensor(qw))


def build_bundle(source):
    full = np.ones(len(source['x']), dtype=bool)
    scaler, anchor = fit_anchor(source, full)
    outer = make_episode(source, full, full, scaler, anchor, 'outer', -1)
    real = []
    for cell in np.unique(source['ids']):
        query = source['ids'] == cell
        inner_scaler, inner_anchor = fit_anchor(source, ~query)
        real.append(make_episode(source, ~query, query, inner_scaler, inner_anchor,
                                 f'physical_{cell}', cell))
    return dict(source=source, scaler=scaler, anchor=anchor, outer=outer, real=real)


def pseudo_episodes(source):
    allocation = pseudo_assign(source)
    if not allocation['available']:
        raise ValueError(allocation['reason'])
    groups = np.asarray(allocation['labels'])
    episodes = []
    for group in np.unique(groups):
        query = groups == group
        scaler, anchor = fit_anchor(source, ~query)
        episodes.append(make_episode(source, ~query, query, scaler, anchor,
                                     f'pseudo_{group}', group))
    return episodes, allocation
