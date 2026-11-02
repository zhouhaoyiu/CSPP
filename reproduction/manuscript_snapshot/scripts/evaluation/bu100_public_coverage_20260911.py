"""Export annotation availability at supplied P times, independent of scores."""
import csv
import json
import h5py
import numpy as np
import bu100_public_baselines_20260911 as base


def main():
    output = []
    for key in ['phasenet_instance','eqtransformer_original']:
        with h5py.File(base.OUT / key / 'annotations.h5') as cache:
            for r in base.records():
                g = cache[r['id']]
                mask = np.asarray(g['published'])
                ref = float(r['reference_p_seconds'])
                p = int(round(ref*100))
                packet = int(np.floor(ref / .5))
                a, b = max(0,int(np.ceil((ref-1.5)*100))), min(len(mask),int(np.floor((ref+1.5)*100))+1)
                published = np.flatnonzero(mask)
                output.append({'id':r['id'],'instrument':r['instrument'],'cohort':r['cohort'],'model':key,
                               'reference_p_seconds':ref,'duration_seconds':len(mask)/100,
                               'reference_sample_published':int(p<len(mask) and mask[p]),
                               'full_reference_packet_published':int(mask[packet*50:(packet+1)*50].all()),
                               'cspp_allowed_at_reference':int(g['allowed'][packet]),
                               'any_sample_within_tolerance_published':int(mask[a:b].any()),
                               'first_published_sample_seconds':published[0]/100 if len(published) else '',
                               'native_windows':json.loads(g.attrs['diagnostics'])['executed_valid_windows']})
    with (base.OUT / 'reference_coverage.csv').open('w',newline='') as f:
        w = csv.DictWriter(f,fieldnames=list(output[0]));w.writeheader();w.writerows(output)


if __name__ == '__main__':
    main()
