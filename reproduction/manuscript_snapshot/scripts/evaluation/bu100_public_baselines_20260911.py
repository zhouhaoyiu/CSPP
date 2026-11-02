"""Frozen native-window causal baselines on every converted BU100 record."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from v20c_public_window_baselines import (
    replay_packet_quality, load_public_model, infer_first_available_annotation,
    extract_closed_candidates, PublicAnnotation,
)

BASE = ROOT / 'outputs/cspp_bu100_20260910'
OUT = ROOT / 'outputs/cspp_bu100_public_baselines_20260911'
LOCAL_CACHE = OUT / 'seisbench_cache'
os.environ['SEISBENCH_CACHE_ROOT'] = str(LOCAL_CACHE)
SOURCE = ROOT / 'experiments/cspp_v27_ncedc_causal_public_baseline_protocol_20260817.json'
GRID = [.001, .003, .005, .01, .02, .03, .05, .075, .1, .15, .2, .3, .4, .5, .6, .7, .8, .9, .95, .975, .99]
TOLS = [1.5, .5, .25]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(8 * 1024**2):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))


def freeze():
    OUT.mkdir(exist_ok=True)
    if (OUT / 'protocol.json').exists():
        raise FileExistsError('Protocol already exists; use infer or score to resume.')
    source = json.loads(SOURCE.read_text())
    hashes = {str(p): sha(p) for p in [SOURCE, BASE / 'waveforms_100hz_zne.h5',
              BASE / 'conversion_status.csv', BASE / 'record_metrics.csv',
              Path(__file__), ROOT / 'src/v20c_public_window_baselines.py', ROOT / 'src/stream_quality_v5.py']}
    for spec in source['models'].values():
        for a in spec['artifacts'].values():
            original = Path(a['path'])
            local = LOCAL_CACHE / 'models/v3' / original.parent.name / original.name
            local.parent.mkdir(parents=True, exist_ok=True)
            if original.is_file():
                import shutil
                shutil.copyfile(original, local)
            elif not local.is_file():
                import requests
                url = f'https://seisbench.gfz.de/mirror/models/v3/{original.parent.name}/{original.name}'
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                local.write_bytes(response.content)
            a['original_path'] = a['path']
            a['path'] = str(local)
            if sha(a['path']) != a['sha256']:
                raise RuntimeError('Public weight or metadata has drifted')
            hashes[a['path']] = a['sha256']
    write_json(OUT / 'protocol.json', {
        'frozen_at': datetime.now(timezone.utc).isoformat(), 'models': source['models'],
        'hashes': hashes, 'threshold_grid': GRID, 'tolerances_seconds': TOLS,
        'cohort': 'All 2136 converted records; no selection by any model result.',
        'matching': 'First available candidate within tolerance; at most one TP per record; unmatched candidates are FP proxies.',
        'input': 'Full common converted 100 Hz ZNE record; only incomplete final packet dropped. No reference-P crop or zero-prefill.',
        'public_semantics': 'Existing native-window first-publication and closed-episode code, weight-specific preprocessing/blinding/stride, valid-packet windows. No CSPP warmup imposed on public candidates.',
        'primary': 'Released public P thresholds, no detection gate; CSPP original 0.1/0.975.',
        'sensitivity': 'Each model scanned on the same cohort. Exploratory not independent testing. CSPP scan already opened. Highest F1 at 1.5 s, ties higher threshold.',
        'deployment': 'Publication clock excludes actual inference time and transmission; this does not measure all optimized online implementations.',
        'units': 'Intensity instrument physical units remain unverified; all methods receive identical numeric input.',
        'device': 'mps', 'threads': 1,
    })
    print('Protocol frozen', flush=True)


def records():
    with (BASE / 'conversion_status.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2136 and all(r['status'] == 'converted' for r in rows)
    return rows


def infer(key, limit):
    protocol = json.loads((OUT / 'protocol.json').read_text())
    spec = protocol['models'][key]
    torch.set_num_threads(1)
    model = load_public_model(spec, device=torch.device('mps'))
    dest = OUT / key
    dest.mkdir(exist_ok=True)
    started = time.monotonic()
    rows = records()[:limit or None]
    with h5py.File(BASE / 'waveforms_100hz_zne.h5', 'r') as source, h5py.File(dest / 'annotations.h5', 'a') as cache:
        for i, r in enumerate(rows):
            rid = r['id']
            if rid in cache and cache[rid].attrs.get('complete', False):
                continue
            if rid in cache:
                del cache[rid]
            wave = np.asarray(source[rid]['wave'], dtype=np.float32)
            wave = wave[:, :wave.shape[1] // 50 * 50]
            valid, allowed, quality = replay_packet_quality(wave, start_epoch=0., warmup_packets=20)
            a = infer_first_available_annotation(model, spec, wave, valid, device=torch.device('mps'))
            g = cache.create_group(rid)
            for name, values in [('p_probability', a.p_probability), ('available_packet', a.available_packet),
                                 ('published', a.published), ('valid', valid), ('allowed', allowed)]:
                g.create_dataset(name, data=values, compression='gzip', compression_opts=1)
            if a.detection_probability is not None:
                g.create_dataset('detection_probability', data=a.detection_probability, compression='gzip', compression_opts=1)
            g.attrs['diagnostics'] = json.dumps(a.diagnostics)
            g.attrs['quality'] = json.dumps(quality)
            g.attrs['wave_sha256'] = hashlib.sha256(wave.tobytes()).hexdigest()
            g.attrs['complete'] = True
            cache.flush()
            if i % 50 == 0 or i + 1 == len(rows):
                print(json.dumps({'model': key, 'records': i + 1, 'total': len(rows), 'seconds': round(time.monotonic() - started, 2)}), flush=True)
    if len(rows) == 2136:
        write_json(dest / 'inference_complete.json', {'records': len(rows), 'seconds_this_session': time.monotonic()-started,
                   'protocol_sha256': sha(OUT / 'protocol.json'), 'annotation_sha256': sha(dest / 'annotations.h5')})


def score(key):
    protocol = json.loads((OUT / 'protocol.json').read_text())
    spec = protocol['models'][key]
    default = spec['operating_points'][0]['p_threshold']
    thresholds = sorted(set(GRID + [default]))
    dest = OUT / key
    fields = ['id', 'cohort', 'instrument', 'model', 'threshold', 'is_default', 'tolerance', 'candidate_count',
              'tp', 'fp_proxy', 'fn', 'signed_error', 'absolute_error', 'availability_latency']
    with (h5py.File(dest / 'annotations.h5', 'r') as cache,
          (dest / 'record_metrics.csv').open('w', newline='') as f,
          (dest / 'default_candidates.jsonl').open('w') as candidates_out):
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records():
            g = cache[r['id']]
            assert g.attrs['complete']
            a = PublicAnnotation(np.asarray(g['p_probability']),
                                 np.asarray(g['detection_probability']) if 'detection_probability' in g else None,
                                 np.asarray(g['available_packet']), np.asarray(g['published']), json.loads(g.attrs['diagnostics']))
            ref = float(r['reference_p_seconds'])
            for threshold in thresholds:
                candidates = extract_closed_candidates(a, method={'p_threshold': threshold}, start_epoch=0.)
                candidates.sort(key=lambda c: c['available_epoch'])
                if threshold == default:
                    candidates_out.write(json.dumps({'id': r['id'], 'candidates': candidates}) + '\n')
                for tol in TOLS:
                    match = next((c for c in candidates if abs(c['estimated_p_epoch'] - ref) <= tol), None)
                    tp = int(match is not None)
                    row = {k: r[k] for k in ['id', 'cohort', 'instrument']}
                    row.update(model=key, threshold=threshold, is_default=int(threshold == default), tolerance=tol,
                               candidate_count=len(candidates), tp=tp, fp_proxy=len(candidates)-tp, fn=1-tp,
                               signed_error=match['estimated_p_epoch']-ref if tp else '',
                               absolute_error=abs(match['estimated_p_epoch']-ref) if tp else '',
                               availability_latency=match['available_epoch']-ref if tp else '')
                    writer.writerow(row)
    print(key, 'scored', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['freeze', 'infer', 'score'])
    p.add_argument('--model', choices=['eqtransformer_original', 'phasenet_instance'])
    p.add_argument('--limit', type=int, default=0)
    args = p.parse_args()
    if args.action == 'freeze':
        freeze()
    elif args.action == 'infer':
        infer(args.model, args.limit)
    else:
        score(args.model)
