"""Verify rebased outputs, original artifacts, selected picks and causal clocks."""
import json

import h5py
import numpy as np
import pandas as pd
import torch

import bu100_public_baselines_20260911 as base
from bu100_rebased_public_20260911 import OUT, PARENT
from public_valid_segment_windows import infer_valid_segment_annotation
from verify_bu100_public_20260911 import numeric_checks


def main():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    for path, expected in protocol['hashes'].items():
        assert base.sha(path) == expected, path
    selected = json.loads((OUT / 'selected.json').read_text())
    results = {'summary_rows': numeric_checks(OUT), 'source_hashes_unchanged': True, 'models': []}
    torch.set_num_threads(1)
    references = {r['id']: float(r['reference_p_seconds']) for r in base.records()}
    with h5py.File(base.BASE / 'waveforms_100hz_zne.h5') as waves:
        for chosen in selected:
            key = chosen['model']
            old_completion = json.loads((PARENT / key / 'inference_complete.json').read_text())
            assert base.sha(PARENT / key / 'annotations.h5') == old_completion['annotation_sha256']
            metrics = pd.read_csv(OUT / key / 'record_metrics.csv')
            best = metrics[metrics.threshold == chosen['threshold']]
            best.to_csv(OUT / key / 'selected_record_metrics.csv', index=False)
            by_id = best[best.tolerance == 1.5].set_index('id')
            no_gap_identical = 0
            pilots = []
            with (h5py.File(OUT / key / 'annotations.h5') as new,
                  h5py.File(PARENT / key / 'annotations.h5') as old,
                  (OUT / key / 'selected_candidates.jsonl').open('w') as picks_out):
                for rid, reference in references.items():
                    g = new[rid]
                    assert g.attrs['complete']
                    for field in ['valid', 'allowed']:
                        np.testing.assert_array_equal(g[field], old[rid][field])
                    if np.asarray(g['valid']).all():
                        for field in ['p_probability', 'available_packet', 'published']:
                            np.testing.assert_array_equal(g[field], old[rid][field])
                        no_gap_identical += 1
                    annotation = base.PublicAnnotation(
                        np.asarray(g['p_probability']),
                        np.asarray(g['detection_probability']) if 'detection_probability' in g else None,
                        np.asarray(g['available_packet']), np.asarray(g['published']), {},
                    )
                    picks = base.extract_closed_candidates(annotation, method={'p_threshold': chosen['threshold']}, start_epoch=0.)
                    picks.sort(key=lambda p: p['available_epoch'])
                    picks_out.write(json.dumps({'id': rid, 'candidates': picks}) + '\n')
                    matched = int(any(abs(p['estimated_p_epoch'] - reference) <= 1.5 for p in picks))
                    assert matched == by_id.loc[rid].tp
                    assert len(picks) - matched == by_id.loc[rid].fp_proxy
                spec = protocol['models'][key]
                cpu = base.load_public_model(spec, device=torch.device('cpu'))
                mps = base.load_public_model(spec, device=torch.device('mps'))
                # Long records ensure every future-perturbation check is nonvacuous.
                for rid in ['r00000', 'r00001', 'r00228']:
                    wave = np.asarray(waves[rid]['wave'], dtype=np.float32)
                    wave = wave[:, :wave.shape[1] // 50 * 50]
                    valid = np.asarray(new[rid]['valid'])
                    original = np.asarray(new[rid]['p_probability'])
                    a = infer_valid_segment_annotation(cpu, spec, wave, valid, device=torch.device('cpu'))
                    diff = float(np.max(np.abs(a.p_probability - original)))
                    assert diff < 2e-4, (key, rid, diff)
                    cut = wave.shape[1] // 100 * 50
                    changed = wave.copy()
                    changed[:, cut:] = .123 - 3 * changed[:, cut:]
                    v2, _, _ = base.replay_packet_quality(changed, start_epoch=0., warmup_packets=20)
                    b = infer_valid_segment_annotation(mps, spec, changed, v2, device=torch.device('mps'))
                    before = ((np.asarray(new[rid]['available_packet']) + 1) * 50 <= cut) & np.asarray(new[rid]['published'])
                    assert before.any(), (key, rid, 'vacuous prefix')
                    np.testing.assert_array_equal(b.p_probability[before], original[before])
                    np.testing.assert_array_equal(b.available_packet[before], np.asarray(new[rid]['available_packet'])[before])
                    pilots.append({'id': rid, 'cpu_mps_max_difference': diff, 'unchanged_prefix_samples': int(before.sum())})
            results['models'].append({'model': key, 'records': len(references),
                                     'original_cache_hash_unchanged': True, 'quality_masks_unchanged': True,
                                     'no_gap_records_bitwise_identical': no_gap_identical, 'pilot_checks': pilots})
    results.update(status='PASS', scope='Code, saved predictions, timing and aggregate consistency; not independent scientific validation.')
    base.write_json(OUT / 'verification.json', results)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
