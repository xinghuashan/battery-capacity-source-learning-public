"""Weighted residual RBF kernel regression and exact candidate ordering."""
import numpy as np

def positive_length_scale(u):
    z=np.asarray(u)
    matrix=np.sum((z[:,None]-z[None])**2,axis=2)/17
    values=matrix[np.triu_indices(len(z),1)];values=values[values>0]
    if not len(values):raise ValueError('degenerate-input: no strictly positive source pair distance')
    return float(np.median(values))


def kernel(a,b,eta,ell):return np.exp(-eta*np.sum((np.asarray(a)[:,None]-np.asarray(b)[None])**2,axis=2)/17/ell)


def choose_grid(rows):return min(rows,key=lambda row:(row['source_macro_query_rmse_pp'],-row['lambda_value'],row['eta']))


def krr_solve(z, t, w, ell, eta, lam, identity=None, role='formal_KRR'):
    matrix = kernel(z, z, eta, ell) + np.diag(lam / np.asarray(w))
    rhs = np.asarray(t)
    factor = np.linalg.cholesky(matrix)
    beta = np.linalg.solve(factor.T, np.linalg.solve(factor, rhs))
    residual = np.linalg.norm(matrix @ beta - rhs) / max(np.linalg.norm(rhs), 1e-30)
    assert residual < 1e-10 and np.isfinite(beta).all()
    return dict(z=np.asarray(z), beta=beta, eta=float(eta), lambda_value=float(lam), ell=float(ell), solve_residual=float(residual))
