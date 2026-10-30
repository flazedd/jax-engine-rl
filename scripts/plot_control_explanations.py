"""Plot configuration levels from verified frozen-checkpoint summaries."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from plotting.style import apply_style, polish, PALETTE

METHODS=['rl2_concat','rl2_hypernet','varibad_concat','varibad_hypernet']
COLORS=[PALETTE['concat'],PALETTE['hyper']]*2

def bars(ax,estimates,ylabel):
 means=np.array([e['mean'] for e in estimates]);ci=np.array([e['ci95'] for e in estimates])
 positions=np.array([-.18,.18,.82,1.18]);width=.34
 ax.bar(positions,means,color=COLORS,width=width,edgecolor='white',linewidth=1.2,zorder=3,
        yerr=[means-ci[:,0],ci[:,1]-means],capsize=4,
        error_kw={'ecolor':'#3a3a3a','elinewidth':1.2})
 for x,m,hi in zip(positions,means,ci[:,1]):
  label=f'{m:.3f}' if means.max()<1 else f'{m:.1f}'
  ax.annotate(label,xy=(x,hi),xytext=(0,6),textcoords='offset points',ha='center',va='bottom',fontsize=8.5,fontweight='bold',color='#333333')
 ax.set(xticks=[0,1],xticklabels=['RL²','VariBAD'],ylabel=ylabel,ylim=(0,ci[:,1].max()*1.22),xlim=(-.55,1.55))
 polish(ax)
 ax.legend(handles=[Patch(facecolor=PALETTE['hyper'],edgecolor='white',label='Hypernet integration'),Patch(facecolor=PALETTE['concat'],edgecolor='white',label='Concat integration')],loc='upper left',frameon=True,facecolor='white',edgecolor='#cccccc',fontsize=8)


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 d=json.loads(a.input.read_text());a.output.mkdir(parents=True,exist_ok=True)
 apply_style()
 plt.rcParams.update({'pdf.fonttype':42,'font.family':'DejaVu Sans'})
 rows={(r['method'],r['seed']):r for r in d['rows']}
 ix=np.random.default_rng(0).integers(20,size=(10000,20))
 def estimate(v):return {'mean':float(v.mean()),'ci95':np.percentile(v[ix].mean(1),[2.5,97.5]).tolist()}
 fig,axes=plt.subplots(1,2,figsize=(11,4.3));levels={}
 for ax,metric,title in zip(axes,['gain','penalty'],['Trading gains (higher is better)','Inventory penalties (lower is better)']):
  estimates=[estimate(np.array([rows[m,s][metric] for s in range(20)])) for m in METHODS]
  levels[metric]=dict(zip(METHODS,estimates));bars(ax,estimates,'Mean episode total');ax.set_title(title)
  if metric=='gain':ax.set_ylim(bottom=50)
 fig.tight_layout();fig.savefig(a.output/'return_components.pdf');plt.close(fig)
 (a.output/'return_component_levels.json').write_text(json.dumps(levels,indent=2)+'\n')
 quality=json.loads((a.input.parent/'decision_quality_comparisons.json').read_text())['common_states']
 fig,ax=plt.subplots(figsize=(9.2,5.8))
 bars(ax,[quality['configurations'][m] for m in METHODS],'Mean decision loss per decision')
 ax.set_ylabel('Mean decision loss per decision (lower is better)')
 fig.tight_layout();fig.savefig(a.output/'decision_quality.pdf');plt.close(fig)
if __name__=='__main__':main()
