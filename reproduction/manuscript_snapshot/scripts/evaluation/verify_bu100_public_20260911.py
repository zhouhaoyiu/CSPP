"""Independent aggregate checks and CPU/causal-prefix checks for BU100."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import torch
import bu100_public_baselines_20260911 as base


def numeric_checks(folder):
    summary = pd.read_csv(folder / 'summary.csv')
    verified = 0
    for model in summary.model.unique():
        records = pd.read_csv(folder / model / 'record_metrics.csv')
        expected = {}
        for group_name, fields in [('all', ['threshold', 'tolerance']),
                                   ('instrument', ['threshold', 'tolerance', 'instrument']),
                                   ('cohort', ['threshold', 'tolerance', 'cohort'])]:
            for key, table in records.groupby(fields, sort=False):
                threshold, tol = key[:2]
                group = 'all' if group_name == 'all' else key[2]
                tp, fp, fn = (int(table[c].sum()) for c in ['tp', 'fp_proxy', 'fn'])
                vals = {'records': len(table), 'tp': tp, 'fp_proxy': fp, 'fn': fn,
                        'precision': tp/(tp+fp) if tp+fp else np.nan,
                        'recall': tp/(tp+fn), 'f1': 2*tp/(2*tp+fp+fn)}
                for column, prefix in [('absolute_error', 'absolute_error'), ('availability_latency', 'availability_latency')]:
                    v = table[column].dropna().to_numpy()
                    vals['median_' + prefix] = float(np.median(v)) if len(v) else np.nan
                    vals['p95_' + prefix] = float(np.quantile(v, .95, method='hazen')) if len(v) else np.nan
                    if column == 'absolute_error':
                        vals['mean_' + prefix] = float(np.mean(v)) if len(v) else np.nan
                expected[(round(threshold, 14), tol, group)] = vals
        for r in summary[summary.model == model].to_dict('records'):
            values = expected[(round(r['threshold'], 14), r['tolerance'], r['group'])]
            for k, v in values.items():
                assert np.isclose(r[k], v, rtol=1e-10, atol=1e-10, equal_nan=True), (model, r['threshold'], k, r[k], v)
            verified += 1
    return verified


def main():
    protocol = json.loads((base.OUT / 'protocol.json').read_text())
    for path, expected in protocol['hashes'].items():
        assert base.sha(path) == expected, path
    results = {'coarse_summary_rows': numeric_checks(base.OUT),
               'refined_summary_rows': numeric_checks(base.OUT / 'refined'),
               'source_hashes_unchanged': True, 'models': []}
    selected = json.loads((base.OUT / 'refined/selected.json').read_text())
    torch.set_num_threads(1)
    references = {r['id']: float(r['reference_p_seconds']) for r in base.records()}
    with (h5py.File(base.BASE / 'waveforms_100hz_zne.h5') as waveforms,
          h5py.File(base.ROOT / 'outputs/cspp_bu100_threshold_scan_20260911/packet_scores.h5') as cspp):
        for chosen in selected:
            model_key = chosen['model']
            spec = protocol['models'][model_key]
            refined = base.OUT / 'refined' / model_key
            metrics = pd.read_csv(refined / 'record_metrics.csv')
            best = metrics[metrics.threshold == chosen['threshold']]
            best.to_csv(refined / 'selected_record_metrics.csv', index=False)
            by_id = best[best.tolerance == 1.5].set_index('id')
            with h5py.File(refined / 'annotations.h5') as annotations, (refined / 'selected_candidates.jsonl').open('w') as output:
                for rid in references:
                    g = annotations[rid]
                    score_cache = np.asarray(cspp[rid]['scores'])
                    np.testing.assert_array_equal(g['valid'], score_cache[3].astype(bool))
                    np.testing.assert_array_equal(g['allowed'], score_cache[4].astype(bool))
                    a = base.PublicAnnotation(np.asarray(g['p_probability']),
                            np.asarray(g['detection_probability']) if 'detection_probability' in g else None,
                            np.asarray(g['available_packet']), np.asarray(g['published']), {})
                    picks = base.extract_closed_candidates(a, method={'p_threshold': chosen['threshold']}, start_epoch=0.)
                    picks.sort(key=lambda x: x['available_epoch'])
                    output.write(json.dumps({'id': rid, 'candidates': picks}) + '\n')
                    # Recompute one-match counts directly from saved pick times.
                    nmatch = int(any(abs(p['estimated_p_epoch']-references[rid]) <= 1.5 for p in picks))
                    row = by_id.loc[rid]
                    assert nmatch == row.tp and len(picks)-nmatch == row.fp_proxy
                cpu = base.load_public_model(spec, device=torch.device('cpu'))
                mps = base.load_public_model(spec, device=torch.device('mps'))
                pilots = []
                for rid in ['r00000', 'r00100', 'r01000']:
                    wave = np.asarray(waveforms[rid]['wave'], dtype=np.float32)
                    wave = wave[:, :wave.shape[1]//50*50]
                    valid = np.asarray(annotations[rid]['valid'])
                    original = np.asarray(annotations[rid]['p_probability'])
                    a = base.infer_first_available_annotation(cpu, spec, wave, valid, device=torch.device('cpu'))
                    diff = float(np.max(np.abs(a.p_probability-original)))
                    assert diff < 2e-4, (model_key,rid,diff)
                    changed = wave.copy()
                    cut = (wave.shape[1]//100)*50
                    changed[:, cut:] = .123 + changed[:, cut:] * -3
                    v2, _, _ = base.replay_packet_quality(changed,start_epoch=0.,warmup_packets=20)
                    changed_a = base.infer_first_available_annotation(mps, spec, changed, v2, device=torch.device('mps'))
                    before = ((np.asarray(annotations[rid]['available_packet'])+1)*50 <= cut) & np.asarray(annotations[rid]['published'])
                    np.testing.assert_array_equal(changed_a.p_probability[before],original[before])
                    pilots.append({'id':rid,'cpu_mps_max_probability_difference':diff,
                                   'unchanged_published_prefix_samples':int(before.sum())})
                results['models'].append({'model':model_key,'records':2136,'selected_threshold':chosen['threshold'],
                                         'quality_masks_equal_cspp':True,'pilot_checks':pilots})
    results['status'] = 'PASS'
    results['scope'] = 'Implementation/aggregate consistency, not independent scientific validation.'
    base.write_json(base.OUT / 'verification.json',results)
    print(json.dumps(results,indent=2))


if __name__ == '__main__':
    main()
