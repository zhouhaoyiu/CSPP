"""Render manuscript figures with GMT from MATLAB-exported statistics."""
from pathlib import Path
import io
import os
import numpy as np
import pandas as pd
import pygmt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get('CSPP_FIGURE_OUT', ROOT / 'outputs/cspp_manuscript_revision_20260911'))
DATA = Path(os.environ.get('CSPP_FIGURE_DATA', OUT / 'figure_data'))
FIG = OUT / 'figures'
COLORS = {'CSPP': '#0072B2', 'PhaseNet': '#D55E00', 'EQTransformer': '#009E73'}
MARKS = {'CSPP': 'c', 'PhaseNet': 't', 'EQTransformer': 's'}


def title(fig, value):
    fig.text(position='TL', text=value, justify='BL', offset='0/0.25c',
             font='9p,Helvetica,black', no_clip=True)


def save(fig, name):
    fig.savefig(FIG / f'{name}.pdf')
    fig.savefig(FIG / f'{name}.png', dpi=400)


def canvas():
    fig = pygmt.Figure()
    fig.shift_origin(yshift='18c')
    return fig


def legend(fig, wide=False):
    spec = '\n'.join(f'S 0.1c {MARKS[m]} 0.18c {COLORS[m]} 0.5p,{COLORS[m]} 0.4c {m}' for m in COLORS)
    if wide:
        spec = 'N 3\n'+spec
        fig.legend(spec=io.StringIO(spec), position='x8c/-1.15c+jTC+w14c', box=False)
        return
    fig.legend(spec=io.StringIO(spec), position='JBC+jTC+o0/0.95c', box=False)


def bars(fig, table, label, groups=('precision', 'recall', 'f1'), names=('Precision', 'Recall', 'F1')):
    fig.basemap(region=[-.5, 2.5, 0, 105], projection='X7.3c/4.6c', frame=['WS', 'ya20f10+lScore (%)', 'xf1'])
    for i, method in enumerate(COLORS):
        row = table[table.method == method].iloc[0]
        fig.plot(x=np.arange(3)+(i-1)*.23, y=[float(row[g])*100 for g in groups],
                 style='b0.19u+b0', fill=COLORS[method], pen='0.3p,white')
    for i, name in enumerate(names):
        fig.text(x=i, y=-8, text=name, font='8.5p,Helvetica', no_clip=True)
    title(fig, label)


def timing(fig, table, label, xfield, yfield):
    fig.basemap(region=[.2, 70, 0, .30], projection='X7.3cl/4.6c',
                frame=['WS', 'xa1f2+lMedian output delay (s)', 'ya.1f.05+lMedian absolute error (s)'])
    for method in COLORS:
        row = table[table.method == method].iloc[0]
        fig.plot(x=[row[xfield]], y=[row[yfield]], style=MARKS[method]+'0.24c', fill=COLORS[method], pen='0.6p,white')
    title(fig, label)


def external():
    table = pd.read_csv(DATA / 'public_event_windows.csv')
    for field, display in [('median_absolute_pick_error_seconds','error_display'),
                           ('median_availability_delay_seconds','delay_display')]:
        table[field] = table[field].fillna(pd.to_numeric(table[display]))
    table = table[~table.dataset.str.contains('GeoNet ', regex=False)].copy()
    table.loc[table.dataset.str.startswith('NCSN'), 'dataset'] = 'NCSN'
    fig = canvas()
    panels = [('precision',118,'xa25+lPrecision (%)','(a) Precision',100),
              ('recall',118,'xa25+lRecall (%)','(b) Recall',100),
              ('f1',118,'xa25+lF1 (%)','(c) F1',100),
              ('median_availability_delay_seconds',36,'xa10+lOutput delay (s)','(d) Median delay',1)]
    for panel,(field,xmax,axis,label,scale) in enumerate(panels):
        if panel:
            fig.shift_origin(xshift='4.35c')
        fig.basemap(region=[0,xmax,-.5,6.3],projection='X3.55c/5.6c',frame=['S',axis])
        title(fig,label)
        for group,dataset in enumerate(['NCSN','TexNet']):
            t = table[table.dataset == dataset].set_index('method')
            header_y = 6.05 if group == 0 else 2.65
            if panel == 0:
                fig.text(x=0,y=header_y,text=f'{dataset} (n = {int(t.records.iloc[0])})',
                         justify='ML',font='8.5p,Helvetica-Bold',no_clip=True)
            for i,method in enumerate(COLORS):
                y = (5.3 if group == 0 else 1.9)-i*.8
                value = float(t.loc[method,field])*scale
                fig.plot(x=[0,xmax],y=[y,y],pen='0.35p,#DDDDDD')
                fig.plot(x=[value],y=[y],style=MARKS[method]+'0.19c',fill=COLORS[method],pen='0.4p,white',no_clip=True)
                fig.text(x=value+xmax*.045,y=y,text=f'{value:.1f}' if panel<3 else f'{value:.2f}',
                         justify='ML',font='8p,Helvetica',no_clip=True)
    fig.shift_origin(xshift='-13.05c')
    legend(fig,wide=True)
    save(fig, 'fig3_external')


def china():
    table = pd.read_csv(DATA / 'china_comparison.csv')
    best = table[(table.setting == 'best') & (table.tolerance == 1.5)]
    fig = canvas()
    fig.basemap(region=[0, 102, 0, 102], projection='X7.1c/5.0c', frame=['WS', 'xa20f10+lRecall (%)', 'ya20f10+lPrecision (%)'])
    cspp = pd.read_csv(DATA / 'china_cspp_all_thresholds.csv')
    fig.plot(x=cspp.recall*100, y=cspp.precision*100, style='c0.07c', fill=COLORS['CSPP'], transparency=60)
    pub = pd.read_csv(DATA / 'china_public_all_thresholds.csv')
    for key, method in [('phasenet_instance','PhaseNet'), ('eqtransformer_original','EQTransformer')]:
        t = pub[pub.model == key].sort_values('threshold')
        fig.plot(x=t.recall*100, y=t.precision*100, style=MARKS[method]+'0.10c',
                 fill=COLORS[method], transparency=20)
    for method in COLORS:
        r = best[(best.method == method) & (best.group == 'all')].iloc[0]
        fig.plot(x=[100*r.recall], y=[100*r.precision], style=MARKS[method]+'0.24c', fill=COLORS[method], pen='0.7p,black')
    title(fig, '(a) Threshold operating points')
    fig.shift_origin(xshift='8.9c')
    fig.basemap(region=[.2, 1.55, 0, 102], projection='X7.1c/5.0c', frame=['WS', 'xa.25+lMatching tolerance (s)', 'ya20f10+lF1 (%)'])
    for method in COLORS:
        t = table[(table.method == method) & (table.setting == 'best') & (table.group == 'all')].sort_values('tolerance')
        fig.plot(x=t.tolerance, y=t.f1*100, pen=f'1.1p,{COLORS[method]}')
        fig.plot(x=t.tolerance, y=t.f1*100, style=MARKS[method]+'0.16c', fill=COLORS[method], pen='0.4p,white')
    title(fig, '(b) Fixed thresholds')
    fig.shift_origin(xshift='-8.9c', yshift='-6.9c')
    for j in (1, 2):
        if j == 2:
            fig.shift_origin(xshift='8.9c')
        region = [-.35,2.35,0,102] if j == 1 else [-1,65,0,102]
        frame = ['WS', 'xf1' if j == 1 else 'xa20f5+lOutput delay (s)', 'ya20f10+lScore (%)' if j == 1 else 'ya20f10+lMatched records (%)']
        fig.basemap(region=region, projection='X7.1c/5.0c', frame=frame)
        for i,method in enumerate(COLORS):
            if j == 1:
                r = best[(best.method == method) & (best.group == 'all')].iloc[0]
                fig.plot(x=np.arange(3)+(i-1)*.09,y=[r.precision*100,r.recall*100,r.f1*100],
                         style=MARKS[method]+'0.18c',fill=COLORS[method],pen='0.4p,white')
            else:
                ecdf = np.loadtxt(DATA / f'china_{method}_ecdf_{j}.txt')
                fig.plot(x=ecdf[:,0],y=ecdf[:,1],pen=f'1.2p,{COLORS[method]}')
        if j == 1:
            for i,label in enumerate(['Precision','Recall','F1']):
                fig.text(x=i,y=-8,text=label,font='9p,Helvetica',no_clip=True)
        title(fig, '(c) Selected operating points' if j == 1 else '(d) Output-delay distribution')
    fig.shift_origin(xshift='-8.9c')
    legend(fig, wide=True)
    save(fig, 'fig4_china')


def continuous():
    table = pd.read_csv(DATA / 'continuous_thresholds_all.csv')
    fig = canvas()
    for i, field in enumerate(['m3_recall', 'm4_recall']):
        if i:
            fig.shift_origin(xshift='9.2c')
        fig.basemap(region=[.0008, 1.2, 0, 105], projection='X7.3cl/4.6c',
                    frame=['WS', 'xa1f2+lDecision threshold', 'ya20f10+lEvent recall (%)'])
        for method in COLORS:
            t = table[table.method == method].sort_values('threshold')
            fig.plot(x=t.threshold, y=t[field]*100, pen=f'1p,{COLORS[method]}',
                     style=MARKS[method]+'0.13c', fill=COLORS[method],no_clip=True)
        title(fig, f'({chr(97+i)}) '+('M >= 3 (82 events)' if i == 0 else 'M >= 4 (13 events)'))
    fig.shift_origin(xshift='-9.2c', yshift='-5.4c')
    fig.basemap(region=[.5, 100, -.5, 2.5], projection='X16.5cl/3.6c',
                frame=['WS', 'xa1f2+lCPU time per 0.5 s update (ms)', 'yf1'])
    table = pd.read_csv(DATA / 'runtime.csv').set_index('method')
    for i, method in enumerate(COLORS):
        r = table.loc[method]
        fig.plot(x=[r.median_ms, r.p95_ms], y=[2-i]*2, pen=f'1.5p,{COLORS[method]}')
        fig.plot(x=[r.median_ms], y=[2-i], style='c0.21c', fill=COLORS[method], pen='0.6p,white')
        fig.plot(x=[r.p95_ms], y=[2-i], style='s0.19c', fill='white', pen=f'1p,{COLORS[method]}')
        # Align method labels with the median/P95 markers while keeping them
        # inside the reserved left side of the log-scale panel.
        fig.text(x=.55, y=2-i, text=method, justify='ML',font='9p,Helvetica',no_clip=True)
        fig.text(x=r.p95_ms*1.08, y=2-i, text=f'{r.median_ms:.3f} / {r.p95_ms:.3f}',
                 justify='ML',font='8p,Helvetica',no_clip=True)
    title(fig, '(c) Single-thread CPU (median / P95)')
    legend(fig, wide=True)
    save(fig, 'fig5_continuous_runtime')


def examples():
    meta = pd.read_csv(DATA / 'examples_metadata.csv')
    fig = canvas()
    for i in range(3):
        if i:
            fig.shift_origin(xshift='-9.2c', yshift='-5.6c')
        wave = np.loadtxt(DATA / f'example{i+1}_wave.txt')
        fig.basemap(region=[-2.5, 4.5, -1.5, 7.5], projection='X7.3c/3.7c', frame=['WS','xa2f1+lTime from reference P (s)','ya3+lDisplay amplitude'])
        keep = (wave[:,0] >= -3) & (wave[:,0] <= 6)
        for k, color in enumerate(['#333333','#0072B2','#D55E00']):
            fig.plot(x=wave[keep,0],y=wave[keep,k+1]+6-3*k,pen=f'0.55p,{color}')
            fig.text(x=-2.35,y=6-3*k+.8,text=['Z','N','E'][k],font='8p,Helvetica',justify='ML')
        fig.plot(x=[0,0],y=[-1.5,7.5],pen='0.8p,black,--')
        r = meta.iloc[i]
        if np.isfinite(r.pick_error):
            fig.plot(x=[r.pick_error]*2,y=[-1.5,7.5],pen='0.8p,#009E73')
            fig.plot(x=[r.release_delay]*2,y=[-1.5,7.5],pen='0.8p,#CC79A7,-.')
        title(fig, f'({chr(97+2*i)}) SNR = {r.snr_db:.1f} dB | '+('matched' if i<2 else 'missed'))
        fig.shift_origin(xshift='9.2c')
        score = np.loadtxt(DATA / f'example{i+1}_scores.txt')
        fig.basemap(region=[-2.5,4.5,0,1.08],projection='X7.3c/3.7c',frame=['WS','xa2f1+lTime from reference P (s)','ya.5f.1+lProbability'])
        for k,color in enumerate(['#0072B2','#D55E00']):
            fig.plot(x=score[:,0],y=score[:,k+1],pen=f'1p,{color}',label=['P detection','P confirmation'][k] if i==2 else None)
        fig.plot(x=[-3,6],y=[.1,.1],pen='0.6p,#0072B2,--')
        fig.plot(x=[-3,6],y=[.975,.975],pen='0.6p,#D55E00,--')
        fig.plot(x=[0,0],y=[0,1.08],pen='0.7p,black,--')
        title(fig, f'({chr(98+2*i)}) Packet scores')
        if i==2:
            fig.shift_origin(xshift='-9.2c')
            spec = 'N 2\nS 0.1c - 0.4c - 1p,#0072B2 0.7c P detection\nS 0.1c - 0.4c - 1p,#D55E00 0.7c P confirmation\n'
            fig.legend(spec=io.StringIO(spec),position='x8.25c/-1.2c+jTC+w12c',box=False)
    save(fig, 'figS1_examples')


def ablation():
    t = pd.read_csv(DATA / 'ablation_all_seeds.csv')
    arms = list(t.arm.unique())
    names = ['Full','No state','No P summary','No timing loss','No spectrum']
    fig = canvas()
    for panel, field in enumerate(['primary_precision','primary_recall','primary_f1']):
        if panel:
            fig.shift_origin(xshift='5.8c')
        scale = 100
        ymax = 105
        fig.basemap(region=[-.5,4.5,0,ymax],projection='X4.7c/4.7c',
                    frame=['WS','xf1', 'ya20f10+l'+['Precision (%)','Recall (%)','F1 (%)'][panel]])
        for i, arm in enumerate(arms):
            rows = t[t.arm == arm].sort_values('seed')
            fig.plot(x=i+np.array([-.13,0,.13]),y=rows[field]*scale,style='c0.16c',fill='#0072B2',pen='0.4p,white')
            fig.text(x=i,y=-ymax*.085,text=names[i],angle=25,justify='RT',font='8p,Helvetica',no_clip=True)
        title(fig, ['(a) Precision','(b) Recall','(c) F1'][panel])
    save(fig, 'figS2_ablation')


if __name__ == '__main__':
    FIG.mkdir(exist_ok=True)
    with pygmt.config(FONT_ANNOT_PRIMARY='9p,Helvetica,black', FONT_LABEL='9p,Helvetica,black',
                      FONT_TITLE='10p,Helvetica-Bold,black', FONT='9p,Helvetica,black',
                      MAP_FRAME_TYPE='plain', MAP_FRAME_PEN='0.6p,black', MAP_TICK_LENGTH_PRIMARY='0.10c',
                      MAP_TICK_PEN_PRIMARY='0.5p,black', MAP_LABEL_OFFSET='0.13c',
                      MAP_ANNOT_OFFSET_PRIMARY='0.08c', PS_MEDIA='A3', PS_CHAR_ENCODING='ISOLatin1+'):
        external()
        china()
        continuous()
        examples()
        ablation()
