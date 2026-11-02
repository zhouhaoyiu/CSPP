"""Rerun both public baselines with label-independent valid-segment origins."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import bu100_public_baselines_20260911 as base
from refine_bu100_public_20260911 import GRIDS
from public_valid_segment_windows import infer_valid_segment_annotation

PARENT = base.OUT
OUT = PARENT / 'valid_segment_origins'


def freeze():
    OUT.mkdir(exist_ok=True)
    if (OUT / 'protocol.json').exists():
        raise FileExistsError('Rebased protocol already frozen')
    protocol = json.loads((PARENT / 'protocol.json').read_text())
    protocol.update(
        frozen_at=datetime.now(timezone.utc).isoformat(),
        parent_protocol_sha256=base.sha(PARENT / 'protocol.json'),
        threshold_grids=GRIDS,
        change='Rebase native windows at the first valid 0.5 s packet of each contiguous valid segment. No label-dependent origin, padding, interpolation, or end-of-record backshift.',
        public_semantics='Full valid native windows, unchanged native stride, blinding, preprocessing, first publication and closed-episode extraction. Absolute record clock retained.',
        selection='Same pre-existing per-model refined grids; maximize F1 on all 2136 records at +/-1.5 s, ties higher threshold. Same-data exploratory optimization.',
    )
    for path in [Path(__file__), base.ROOT / 'src/public_valid_segment_windows.py']:
        protocol['hashes'][str(path)] = base.sha(path)
    base.write_json(OUT / 'protocol.json', protocol)
    print('Valid-segment-origin protocol frozen', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'infer', 'score', 'coverage'])
    parser.add_argument('--model', choices=list(GRIDS))
    args = parser.parse_args()
    if args.action == 'freeze':
        freeze()
    else:
        base.OUT = OUT
        base.infer_first_available_annotation = infer_valid_segment_annotation
        if args.action == 'coverage':
            import bu100_public_coverage_20260911 as coverage
            coverage.main()
        elif args.action == 'infer':
            base.infer(args.model, 0)
        else:
            base.GRID = GRIDS[args.model]
            base.score(args.model)
