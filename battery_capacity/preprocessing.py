"""Record-order and charge-phase checks for the six development cells."""
from collections import Counter
from datetime import datetime
from pathlib import Path
import hashlib
import json
import re
import numpy as np
import pandas as pd
from scipy.io import loadmat
from .features import phase_audit

FIELDS = ('cycle', 'q', 'event', 'reference_raw_record', 'feature_raw_record')


def admit_clock_scope(path, cell, report, rows, stages, inventory, output):
    """Recheck only the original, fingerprint-bound prefit RW clock admission."""
    config_path = Path(__file__).resolve().parents[1] / 'config/accepted_clock_scope.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    identity = config['cells'][path.name]
    assert report['raw_sha256'] == identity['raw_sha256'], 'Original raw fingerprint changed'
    assert len(cell['q']) == identity['eligible_events']
    exceptions = [e for e in config['allowed_documented_clock_exceptions'] if e['cell'] == path.name]
    expected = [{'raw_record': e['raw_record'], 'failure': 'Explicit wallclock timestamp reversal'} for e in exceptions]
    assert report['violations'] == expected, 'Unadmitted audit violation'
    assert not any(t['across_available_records_reversal'] for t in inventory)
    assert [t['raw_record'] for t in inventory if t['within_record_reversal']] == [e['raw_record'] for e in exceptions]
    diagnostics = []
    for exception in exceptions:
        record = exception['raw_record']
        row = rows[record-1]
        times = np.asarray(row['system_time']).astype(str)
        valid = np.flatnonzero(~np.isin(np.char.lower(np.char.strip(times)), ('nan', '', 'nat')))
        parsed = pd.to_datetime(times[valid], format='%Y-%m-%d,%H:%M:%S').to_numpy()
        differences = np.diff(parsed) / np.timedelta64(1, 's')
        jumps = np.flatnonzero(differences < 0)
        assert len(jumps) == 1
        index = int(jumps[0])
        before, after = int(valid[index]), int(valid[index+1])
        assert before == exception['index_before_zero_based'] and after == exception['index_after_zero_based']
        assert float(differences[index]) == exception['delta_seconds']
        native_delta = float(row['relative_time_min'][after]-row['relative_time_min'][before])
        assert native_delta > 0
        feature = bool(np.any(cell['feature_raw_record'] == record))
        target = bool(np.any(cell['reference_raw_record'] == record))
        assert feature == exception['selected_feature'] and target == exception['selected_target']
        if feature:
            selected = [s for s in stages if s['eligible'] and s['feature_raw_record'] == record]
            assert selected
            for stage in selected:
                assert not {before, after} & (set(stage['cc_indices']) | set(stage['cv_indices']))
        diagnostics.append(dict(**exception, relative_time_delta_min=native_delta))
    receipt = dict(admitted=True, scope=config['scope'], original_audit_passed=report['passed'],
                   audit_sha256=sha(output/(path.stem+'_audit.json')),
                   original_amendment_id=config['original_amendment_id'],
                   original_amendment_sha256=config['original_amendment_sha256'],
                   reduced_config_sha256=sha(config_path), raw_sha256=report['raw_sha256'],
                   complete_violation_set_exact_match=True, diagnostics=diagnostics)
    json_save(output/(path.stem+'_clock_admission.json'), receipt)
    return receipt

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timestamp(value):
    if str(value).strip().lower() in ('nan', '', 'nat'):
        return None
    return datetime.strptime(str(value), '%Y-%m-%d,%H:%M:%S')

def json_save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def audit_cell(path, output, cached=None):
    print(json.dumps({'audit_read': path.name}), flush=True)
    mat = loadmat(path, simplify_cells=True)  # one load per file
    rows = mat['data']
    capacity = np.asarray(mat['summary']['discharge_capacity_Ah'], dtype=float)
    descs = [r['description'].strip() for r in rows]
    is_rw = path.name.startswith('RW_')
    expected = ('1C' if is_rw else '2C')+' charge and 1C discharge [test capacity]'
    routine_pattern = r'(?:0.5C|3C) charge and random discharge' if is_rw else r'2C charge and [0-9.]+C discharge to 3V'
    refs, stages, entries, excluded, violations = [], [], [], [], []
    event, routine, last = 0, 0, None
    time_inventory, prior_last_time = [], None
    for j, row in enumerate(rows):
        desc = descs[j]
        raw_times = np.asarray(row['system_time']).astype(str)
        missing = np.isin(np.char.lower(np.char.strip(raw_times)), ('nan', '', 'nat'))
        parsed = pd.to_datetime(raw_times[~missing], format='%Y-%m-%d,%H:%M:%S')
        valid_times = parsed.to_numpy()
        descending = bool(np.any(np.diff(valid_times) < np.timedelta64(0, 's')))
        across_descending = bool(len(valid_times) and prior_last_time is not None and valid_times[0] < prior_last_time)
        equal_across = bool(len(valid_times) and prior_last_time is not None and valid_times[0] == prior_last_time)
        if descending or across_descending:
            violations.append({'raw_record': j+1, 'failure': 'Explicit wallclock timestamp reversal'})
        time_inventory.append({'raw_record': j+1, 'samples': len(raw_times), 'missing_indices': np.flatnonzero(missing).tolist(),
                               'valid_count': len(valid_times), 'within_record_reversal': descending,
                               'across_available_records_reversal': across_descending,
                               'equal_adjacent_valid_timestamps': int(np.sum(np.diff(valid_times) == np.timedelta64(0, 's'))),
                               'equal_across_record_timestamp': equal_across})
        if len(valid_times):
            prior_last_time = valid_times[-1]
        if 'test capacity' in desc.lower():
            known_initial = not is_rw and j == 0 and desc == '0.5C charge and 0.2C discharge [test capacity]'
            target_event = desc == expected and (is_rw or last is not None)
            i, v, q, t = [np.asarray(row[k], dtype=float) for k in
                          ('current_A', 'voltage_V', 'capacity_Ah', 'relative_time_min')]
            pos, neg, rest = np.flatnonzero(i > .1), np.flatnonzero(i < -.1), np.flatnonzero(np.abs(i) < .01)
            if not len(pos) or not len(neg):
                violations.append({'raw_record': j+1, 'failure': 'Reference missing charge/discharge'})
                if target_event:
                    event += 1
                continue
            # CC precedes the voltage-clamped tail; do not summarize CV as CC.
            cc_platform = pos[v[pos] < 4.18]
            charge_A = float(np.median(i[cc_platform])) if len(cc_platform) >= 3 else None
            discharge_A = float(np.median(i[neg]))
            ref = {'audit_reference_index': len(refs), 'event': event if target_event else None,
                   'target_event': target_event, 'known_initial_descriptive_only': known_initial,
                   'raw_record': j+1, 'description': desc, 'cc_platform_points': int(len(cc_platform)),
                   'cc_platform_indices': cc_platform.tolist(),
                   'charge_plateau_median_A': charge_A, 'charge_max_V': float(v[pos].max()),
                   'charge_end_positive_A': float(i[pos[-1]]), 'charge_end_positive_V': float(v[pos[-1]]),
                   'discharge_plateau_median_A': discharge_A, 'discharge_end_V': float(v[neg[-1]]),
                   'counter_end_Ah': float(q[neg[-1]]), 'summary_capacity_Ah': float(capacity[j]),
                   'counter_abs_error_Ah': float(abs(q[neg[-1]]-capacity[j])),
                   'rest_samples': int(len(rest)), 'rest_span_min': float(t[rest[-1]]-t[rest[0]]) if len(rest) else 0,
                   'temperature_available': 'temperature_C' in row, 'reference_start': str(row['system_time'][0]),
                   'anomalies': []}
            if 'temperature_C' in row:
                temp = np.asarray(row['temperature_C'], dtype=float)
                ref.update(temperature_min_C=float(temp.min()), temperature_mean_C=float(temp.mean()), temperature_max_C=float(temp.max()))
            if desc != expected and not known_initial:
                ref['anomalies'].append('Unexpected reference protocol category')
            if charge_A is None:
                ref['anomalies'].append('Reference CC platform not confirmed')
            elif not known_initial and abs(charge_A-(2 if is_rw else 4)) > (.1 if is_rw else .2):
                ref['anomalies'].append('Reference charge plateau outside +/-5%')
            if not known_initial and abs(discharge_A+2) > .1:
                ref['anomalies'].append('Reference discharge plateau outside +/-5%')
            if abs(ref['discharge_end_V']-2.5) > .02:
                ref['anomalies'].append('Reference discharge cutoff outside +/-0.02V')
            if ref['counter_abs_error_Ah'] > 1e-8:
                ref['anomalies'].append('Capacity counter mismatch >1e-8Ah')
            refs.append(ref)
            if ref['anomalies']:
                violations.append({'raw_record': j+1, 'failures': ref['anomalies']})
            if target_event and last is not None:
                x, stage = phase_audit(rows[last], 6 if is_rw else 4)
                stage.update(event=event, feature_raw_record=last+1, reference_raw_record=j+1,
                             cycle=routine, description=descs[last], input_end=str(rows[last]['system_time'][-1]),
                             reference_start=str(row['system_time'][0]))
                input_time, ref_time = timestamp(stage['input_end']), timestamp(stage['reference_start'])
                stage['timestamp_pair_available'] = input_time is not None and ref_time is not None
                stage['raw_record_order_verified'] = last < j
                stage['equal_endpoint_timestamps'] = bool(stage['timestamp_pair_available'] and input_time == ref_time)
                if stage['timestamp_pair_available'] and input_time > ref_time:
                    stage['failures'].append('Input timestamp explicitly after reference')
                stages.append(stage)
                if stage['eligible']:
                    entries.append(dict(x=x, q=capacity[j]/2, cycle=routine, event=event,
                                        feature_raw_record=last+1, reference_raw_record=j+1))
                    if stage['failures']:
                        violations.append({'event': event, 'feature_raw_record': last+1, 'failures': stage['failures']})
                else:
                    excluded.append(event)
            elif target_event:
                excluded.append(event)
            if target_event:
                event += 1
        elif re.fullmatch(routine_pattern, desc):
            routine += 1
            last = j
        else:
            violations.append({'raw_record': j+1, 'failure': 'Unknown non-reference description', 'description': desc})
    cell = {'cell': path.name, 'x': np.array([r['x'] for r in entries]),
            **{k: np.array([r[k] for r in entries]) for k in FIELDS}}
    if cached is not None:
        for k in ('x',)+FIELDS:
            if not np.array_equal(cell[k], cached[k], equal_nan=True):
                violations.append({'failure': 'Cached R3 mismatch', 'field': k})
    np.savez(output/(path.stem+'_observations.npz'), **{k: v for k, v in cell.items() if k != 'cell'})
    json_save(output/(path.stem+'_timestamp_inventory.json'), time_inventory)
    json_save(output/(path.stem+'_stages.json'), stages)
    pd.DataFrame(refs).to_csv(output/(path.stem+'_references.csv'), index=False)
    history = {'past_load': 'current_A waveform of previous records available before reference',
               'rest': 'zero-current samples and system_time observed; no missing between-record history assumed',
               'temperature': 'per-sample temperature_C observed; no ambient/control setting supplied',
               'fields': list(rows[0]), 'system_time': 'timestamp strings YYYY-MM-DD,HH:MM:SS',
               'relative_time_min': 'record-relative; retained raw rest samples'}
    report = {'cell': path.name, 'raw_sha256': sha(path), 'read_path': str(path), 'raw_records': len(rows),
              'descriptions': dict(Counter(descs)), 'reference_events': event, 'all_reference_records': len(refs), 'eligible_events': len(entries),
              'excluded_input_events': excluded, 'nonfinite_labels': cell['event'][~np.isfinite(cell['q'])].tolist(),
              'counter_max_abs_error_Ah': max(r['counter_abs_error_Ah'] for r in refs),
              'reference_cycle_gaps': np.diff([s['cycle'] for s in stages]).tolist(), 'history_availability': history,
              'timestamp_missing_samples': sum(len(t['missing_indices']) for t in time_inventory),
              'timestamp_missing_raw_records': [t['raw_record'] for t in time_inventory if t['missing_indices']],
              'wallclock_unknown_input_pairs': [s['event'] for s in stages if not s['timestamp_pair_available']],
              'violations': violations, 'passed': not violations}
    json_save(output/(path.stem+'_audit.json'), report)
    # Preserve the original failed wallclock audit; admission has its own receipt.
    report['scope_admitted'] = admit_clock_scope(path, cell, report, rows, stages, time_inventory, output)['admitted']
    print(json.dumps({'audit_complete': path.name, 'eligible': len(entries), 'violations': len(violations)}), flush=True)
    return cell, report
