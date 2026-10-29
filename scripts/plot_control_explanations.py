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
 fig,ax=plt.subplots(figsize=(8,3.2))
 for i,(key,label) in enumerate([('rl2','RL²'),('varibad','VariBAD')]):
  c=d['comparisons'][key]
  for metric,offset,marker,scale in [('loss',-.12,'o',128),('matched_loss_per_step',.12,'s',1)]:
   e=c[metric];v=e['mean']/scale;lo,hi=np.array(e['ci95'])/scale
   ax.errorbar(v,i+offset,xerr=[[v-lo],[hi-v]],fmt=marker,color='black',mfc='white' if marker=='s' else 'black',capsize=4,label=('Own trajectories' if marker=='o' else 'Matched states') if i==0 else None)
 ax.axvline(0,color='grey',lw=1);ax.set(yticks=[0,1],yticklabels=['RL²','VariBAD'],xlabel='Decision loss difference per decision\n(hypernetwork minus concatenation)');ax.invert_yaxis();ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False,loc='lower left',bbox_to_anchor=(0,1.01),ncol=2);fig.tight_layout();fig.savefig(a.output/'decision_quality.pdf');plt.close(fig)
if __name__=='__main__':main()
