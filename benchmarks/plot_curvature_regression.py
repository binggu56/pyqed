"""Plot same-node repeated kernel comparisons and standalone curvature controls."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('directory',type=Path)
p.add_argument('--serial',type=Path,help='Optional one-worker summary.json')
a=p.parse_args()
data=json.loads((a.directory/'summary.json').read_text())
reports=data['reports']; variants=['baseline','current','fixed']
colors=['#8996A8','#C77C42','#168392']
fig,axes=plt.subplots(1,3,figsize=(11,3.7),layout='constrained')
metrics=[('First-integral time (s)',lambda r:r['stage_seconds']['first']),
         ('Curvature-integral time (s)',lambda r:r['stage_seconds']['curvature']),
         ('Curvature peak RSS (MiB)',lambda r:r['stage_peak_rss']['curvature']/2**20)]
summary={}
for ax,(label,getter) in zip(axes,metrics):
    for i,(variant,color) in enumerate(zip(variants,colors)):
        values=[getter(reports[f'{variant}-{j}']) for j in range(2)]
        median=float(np.median(values))
        ax.bar(i,median,color=color,width=.6)
        ax.plot([i-.08,i+.08],values,'o',color='#202832',markersize=3)
        ax.text(i,max(values)*1.025,f'{median:.2f}',ha='center',fontsize=9)
        summary.setdefault(variant,{})[label]=median
    ax.set(xticks=range(3),xticklabels=['Baseline','Current','Fixed'],ylabel=label)
    ax.set_ylim(0,max(getter(reports[f'{v}-{j}']) for v in variants for j in range(2))*1.16)
    ax.spines[['top','right']].set_visible(False)
fig.suptitle(f"Pyrazine/cc-pVDZ integral stages: {data['provenance']['host']}, {data['provenance']['cpus']} workers\n22 dimensionless modes, 63 pairs; synthetic sensitivity; dots show two runs",fontsize=11)
for ext in ('png','pdf'): fig.savefig(a.directory/f'curvature_regression.{ext}',dpi=300)
fig,axes=plt.subplots(1,2,figsize=(8,3.5),layout='constrained')
for ax,(label,getter) in zip(axes,metrics[1:]):
    for i,variant in enumerate(['current','fixed']):
        paired=np.median([getter(reports[f'{variant}-{j}']) for j in range(2)])
        alone=getter(reports[f'{variant}-alone'])
        ax.bar(i-.16,paired,.32,color=colors[0],label='After first response' if i==0 else None)
        ax.bar(i+.16,alone,.32,color=colors[2],label='Curvature only' if i==0 else None)
    ax.set(xticks=[0,1],xticklabels=['Current','Fixed'],ylabel=label)
    ax.spines[['top','right']].set_visible(False)
    ax.set_ylim(0,ax.get_ylim()[1]*1.3)
axes[0].legend(frameon=False,fontsize=9)
fig.suptitle('Separate-process controls for first-response memory retention',fontsize=11)
for ext in ('png','pdf'): fig.savefig(a.directory/f'curvature_retention.{ext}',dpi=300)
(a.directory/'plot_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
if a.serial:
    serial=json.loads(a.serial.read_text())['reports']
    fig,axes=plt.subplots(1,3,figsize=(11,3.7),layout='constrained')
    for ax,(label,getter) in zip(axes,metrics):
        values=[getter(serial[v]) for v in variants]
        ax.bar(range(3),values,color=colors,width=.6)
        for i,value in enumerate(values):
            ax.text(i,value*1.025,f'{value:.2f}',ha='center',fontsize=9)
        ax.set(xticks=range(3),xticklabels=['Baseline','Current','Fixed'],
               ylabel=label,ylim=(0,max(values)*1.16))
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Pyrazine/cc-pVDZ integral stages: one worker, one run per variant\n'
                 '22 dimensionless modes, 63 pairs; synthetic sensitivity',fontsize=11)
    for ext in ('png','pdf'):
        fig.savefig(a.directory/f'curvature_serial.{ext}',dpi=300)
