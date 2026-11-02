"""Post-coarse refinement; cache reuse, without modifying coarse outputs."""
import argparse
import json
from pathlib import Path
import bu100_public_baselines_20260911 as base

COARSE = base.OUT
REFINED = COARSE / 'refined'
GRIDS = {
    'phasenet_instance': sorted(set(base.GRID + [i / 1000 for i in range(175, 426, 25)])),
    'eqtransformer_original': sorted(set(base.GRID + [1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, .0001, .0003, .0005, .0007])),
}


def freeze():
    REFINED.mkdir(exist_ok=True)
    if (REFINED / 'protocol.json').exists():
        raise FileExistsError('Refinement already frozen')
    protocol = json.loads((COARSE / 'protocol.json').read_text())
    protocol.update(refinement='After inspecting coarse results: extend EQT lower boundary, refine PhaseNet interior maximum.',
                    parent_protocol_sha256=base.sha(COARSE / 'protocol.json'),
                    parent_summary_sha256=base.sha(COARSE / 'summary.csv'),
                    threshold_grids=GRIDS, refinement_code_sha256=base.sha(__file__))
    base.write_json(REFINED / 'protocol.json', protocol)
    for key in GRIDS:
        dest = REFINED / key
        dest.mkdir(exist_ok=True)
        (dest / 'annotations.h5').symlink_to(COARSE / key / 'annotations.h5')
    print('Refinement frozen')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'score'])
    parser.add_argument('--model', choices=list(GRIDS))
    args = parser.parse_args()
    if args.action == 'freeze':
        freeze()
    else:
        base.OUT = REFINED
        base.GRID = GRIDS[args.model]
        base.score(args.model)
