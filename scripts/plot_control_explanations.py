"""Charts of saved return components and matched decision loss; no model updates."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();d=json.loads(a.input.read_text());a.output.mkdir(parents=True,exist_ok=True)
 plt.rcParams.update({'font.size':12,'pdf.fonttype':42,'font.family':'DejaVu Sans'})
 rows={(r['method'],r['seed']):r for r in d['rows']}
 ix=np.random.default_rng(0).integers(20,size=(10000,20))
 comparisons=[('rl2_hypernet','rl2_concat','RL²: hypernet − concat.'),('varibad_hypernet','varibad_concat','VariBAD: hypernet − concat.'),('rl2_concat','varibad_concat','Concatenation: RL² − VariBAD'),('rl2_hypernet','varibad_hypernet','Hypernetwork: RL² − VariBAD')]
 fig,axes=plt.subplots(2,2,figsize=(11,6.5),sharex=True);estimates={}
 for ax,(first,second,label) in zip(axes.flat,comparisons):
  c={}
  for metric in ['gain','penalty','return']:
   values=np.array([rows[first,seed][metric]-rows[second,seed][metric] for seed in range(20)]);c[metric]={'mean':float(values.mean()),'ci95':np.percentile(values[ix].mean(1),[2.5,97.5]).tolist()}
  estimates[first+' minus '+second]=c
  for y,metric,sign in [(0,'gain',1),(1,'penalty',-1)]:
   e=c[metric];v=sign*e['mean'];lo,hi=sorted(sign*x for x in e['ci95']);ax.barh(y,v,height=.45,color=['#b4bec4','#269d91'][y]);ax.errorbar(v,y,xerr=[[v-lo],[hi-v]],fmt='none',color='black',capsize=4)
  ax.barh(2,c['return']['mean'],height=.45,color='#303030');ax.axvline(0,color='black',lw=.8);ax.set(yticks=[0,1,2],yticklabels=['Trading gains','Penalty savings','Total return gain'],title=label);ax.invert_yaxis();ax.spines[['top','right']].set_visible(False)
 for ax in axes[-1]:ax.set_xlabel('Difference in mean episode totals')
 fig.tight_layout();fig.savefig(a.output/'return_components.pdf');plt.close(fig)
 (a.output/'return_component_comparisons.json').write_text(json.dumps(estimates,indent=2)+'\n')
 quality=json.loads((a.input.parent/'decision_quality_comparisons.json').read_text())
 fig,axes=plt.subplots(2,2,figsize=(11,6),sharex=True)
 for ax,c in zip(axes.flat,quality['comparisons']):
  for y,key,marker in [(0,'all_decisions','o'),(1,'comparable_situations','s')]:
   e=c[key];v=e['mean'];lo,hi=e['ci95'];ax.errorbar(v,y,xerr=[[v-lo],[hi-v]],fmt=marker,color='black',mfc='white' if marker=='s' else 'black',capsize=4,ms=6)
  ax.axvline(0,color='grey',lw=1);ax.set(yticks=[0,1],yticklabels=['All recorded decisions','Comparable situations'],title=c['label'],ylim=(1.6,-.6));ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0)
 for ax in axes[-1]:ax.set_xlabel('Reduction in decision loss per decision\n(positive favours the first named agent)')
 fig.tight_layout();fig.savefig(a.output/'decision_quality.pdf');plt.close(fig)
if __name__=='__main__':main()
