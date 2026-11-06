"""Charge-window features and strictly preceding ordinary-cycle history."""
import re
import numpy as np
from scipy.stats import entropy, kurtosis, skew

KEYS = ['voltage mean', 'voltage std', 'voltage kurtosis', 'voltage skewness',
        'CC Q', 'CC charge time', 'voltage entropy',
        'current mean', 'current std', 'current kurtosis', 'current skewness',
        'CV Q', 'CV charge time', 'current entropy']


def normalized_entropy(values):
    span = np.ptp(values)
    return float(entropy((values-values.min())/span)) if span > 0 else np.nan


def extract(cycle, upper=4.195):
    v, i, t, q = [np.asarray(cycle[key], dtype=float) for key in
                   ('voltage_V', 'current_A', 'relative_time_min', 'capacity_Ah')]
    # Positive charge current identifies charging independently of capacity target.
    cc = np.flatnonzero((v >= 4.) & (v <= upper) & (i >= 3.9))
    cv = np.flatnonzero((i >= .1) & (i <= .5))
    if len(cc) < 3 or len(cv) < 3:
        return np.full(len(KEYS), np.nan)
    a, b = v[cc], i[cv]
    cc_time, cv_time = 60*(t[cc[-1]]-t[cc[0]]), 60*(t[cv[-1]]-t[cv[0]])
    if cc_time <= 0 or cv_time <= 0:
        return np.full(len(KEYS), np.nan)
    return np.array([a.mean(), a.std(), kurtosis(a), skew(a), q[cc[-1]]-q[cc[0]],
                     cc_time, normalized_entropy(a),
                     b.mean(), b.std(), kurtosis(b), skew(b), q[cv[-1]]-q[cv[0]],
                     cv_time, normalized_entropy(b)])

RAW_HISTORY = ['discharge_abs_current_mean_A', 'discharge_abs_current_std_A',
           'discharge_duration_s', 'within_record_rest_duration_s',
           'discharge_temperature_mean_C']


HISTORY_INDEX = [0, 1, 2, 4]


HISTORY = [RAW_HISTORY[j] for j in HISTORY_INDEX]


DROP = [j for j, k in enumerate(KEYS) if k not in ('CC Q', 'CC charge time')]


ROUTINE = re.compile(r'(?:0.5C|3C) charge and random discharge')


def ordinary(row):
    return bool(ROUTINE.fullmatch(str(row['description']).strip()))


def summary(row):
    """Integrate only within positive-time intervals of the same native phase.

    Reset boundaries and charge/discharge/rest transitions contribute no duration.
    Piecewise linear current gives exact first/second moments on each interval.
    No voltage, capacity, power, system_time or summary table is consulted.
    """
    assert ordinary(row), 'Reference/unknown waveform access forbidden'
    current = np.asarray(row['current_A'], dtype=float)
    time = np.asarray(row['relative_time_min'], dtype=float)
    temperature = np.asarray(row['temperature_C'], dtype=float)
    assert current.ndim == time.ndim == temperature.ndim == 1
    assert len(current) == len(time) == len(temperature)
    assert np.isfinite(current).all() and np.isfinite(time).all() and np.isfinite(temperature).all()
    dt = 60*np.diff(time)
    discharge = (dt > 0) & (current[:-1] < -.1) & (current[1:] < -.1)
    rest = (dt > 0) & (abs(current[:-1]) < .01) & (abs(current[1:]) < .01)
    duration = float(dt[discharge].sum())
    assert duration > 0, 'Ordinary cycle has no measurable discharge duration'
    a, b, d = -current[:-1][discharge], -current[1:][discharge], dt[discharge]
    mean = float(np.sum(d*(a+b)/2)/duration)
    second = float(np.sum(d*(a*a+a*b+b*b)/3)/duration)
    temp = float(np.sum(d*(temperature[:-1][discharge]+temperature[1:][discharge])/2)/duration)
    value = np.array([mean, np.sqrt(max(0., second-mean*mean)), duration, float(dt[rest].sum()), temp])
    return value, {'discharge_intervals': int(discharge.sum()), 'rest_intervals': int(rest.sum()),
                   'relative_time_reset_count': int((dt < 0).sum())}


def history_for(rows, feature_record, width):
    """Raw record is one-based; the current feature record is excluded in full."""
    candidates = [(j+1, row) for j, row in enumerate(rows[:feature_record-1]) if ordinary(row)]
    past = candidates[-width:]
    assert len(past) == width, 'Do not silently shorten history'
    return np.mean([summary(row)[0] for _, row in past], axis=0), [j for j, _ in past]

def independent_features(row, cc, cv):
    """Scalar moments reproduce scipy's biased, Fisher statistical conventions."""
    def stats(a):
        values = [float(z) for z in a]
        n = len(values)
        mean = sum(values)/n
        m2 = sum((z-mean)**2 for z in values)/n
        m3 = sum((z-mean)**3 for z in values)/n
        m4 = sum((z-mean)**4 for z in values)/n
        minimum = min(values)
        span = max(values)-minimum
        mass = [(z-minimum)/span for z in values]
        total = sum(mass)
        ent = -sum((z/total)*np.log(z/total) for z in mass if z > 0)
        return [mean, np.sqrt(m2), m4/m2**2-3, m3/m2**1.5], ent
    v, i, t, q = [np.asarray(row[k], dtype=float) for k in
                  ('voltage_V', 'current_A', 'relative_time_min', 'capacity_Ah')]
    a, ae = stats(v[cc])
    b, be = stats(i[cv])
    return np.array(a+[q[cc[-1]]-q[cc[0]], 60*(t[cc[-1]]-t[cc[0]]), ae]
                    +b+[q[cv[-1]]-q[cv[0]], 60*(t[cv[-1]]-t[cv[0]]), be])


def phase_audit(row, plateau):
    v, i, t, q = [np.asarray(row[k], dtype=float) for k in
                  ('voltage_V', 'current_A', 'relative_time_min', 'capacity_Ah')]
    cc = np.flatnonzero((v >= 4) & (v <= 4.195) & (i >= 3.9))
    cv = np.flatnonzero((i >= .1) & (i <= .5))
    feature = extract(row)
    result = {'eligible': bool(np.isfinite(feature).all()), 'cc_indices': cc.tolist(),
              'cv_indices': cv.tolist(), 'failures': []}
    if not result['eligible']:
        return feature, result
    failures = result['failures']
    cs, vs = slice(cc[0], cc[-1]+1), slice(cv[0], cv[-1]+1)
    frac = float(np.mean((i[cc] >= plateau*.95) & (i[cc] <= plateau*1.05)))
    result['cc_plateau_fraction'] = frac
    if frac < .95:
        failures.append('CC plateau fraction below .95')
    if np.any(i[cs] <= 0) or np.any(np.diff(t[cs]) < 0) or np.any(np.diff(q[cs]) < -1e-8):
        failures.append('CC crosses nonpositive charge/time/counter restart')
    if cv[0] <= cc[-1] or np.any((v[cv] < 4.18) | (v[cv] > 4.22)):
        failures.append('CV not after CC at 4.18-4.22V')
    if np.any(i[vs] <= 0) or np.any(np.diff(t[vs]) < 0) or np.any(v[vs] < 4.18) or np.any(i[vs] > .55):
        failures.append('CV crosses discharge/time restart/low voltage/new high current phase')
    if i[cv[-1]] > i[cv[0]]:
        failures.append('CV final current above initial current')
    recomputed = independent_features(row, cc, cv)
    result['recompute_max_abs_error'] = float(np.max(np.abs(recomputed-feature)))
    result['recompute_errors'] = np.abs(recomputed-feature).tolist()
    if not np.allclose(recomputed, feature, rtol=1e-9, atol=1e-10):
        failures.append('Independent 14-feature mismatch')
    result.update({'cc_first_time_min': float(t[cc[0]]), 'cc_last_time_min': float(t[cc[-1]]),
                   'cv_first_time_min': float(t[cv[0]]), 'cv_last_time_min': float(t[cv[-1]]),
                   'cv_first_A': float(i[cv[0]]), 'cv_last_A': float(i[cv[-1]]),
                   'cv_voltage_min_V': float(v[cv].min()), 'cv_voltage_max_V': float(v[cv].max())})
    return feature, result
