"""All four frozen agents after natural switches, using verified saved summaries."""
import argparse,json,tarfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True);p.add_argument('--decision-loss',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 with tarfile.open(a.archive) as t:n=json.load(t.extractfile('analysis/natural_switch_evaluation.json'))
 loss=json.loads(a.decision_loss.read_text());nr={(r['method'],r['seed']):r for r in n['rows']};lr={(r['method'],r['seed']):r for r in loss['rows']};assert len(nr)==len(lr)==80
 for key,row in lr.items():
  np.testing.assert_allclose(row['return'],nr[key]['mean_episode_return'],atol=1e-10,rtol=0);assert row['identity']['checkpoint']==nr[key]['checkpoint_sha256']
 methods=[('rl2_concat','RL² concat.','#3b5967','--','o'),('rl2_hypernet','RL² hypernet','#3b5967','-','o'),('varibad_concat','VariBAD concat.','#269d91','--','s'),('varibad_hypernet','VariBAD hypernet','#269d91','-','s')]
 plt.rcParams.update({'font.size':11,'pdf.fonttype':42});fig,axes=plt.subplots(1,3,figsize=(12,4.3));ix=np.random.default_rng(0).integers(20,size=(10000,20));labels=['0','1–4','5–9','10–19','20–39','40–79','80–126']
 for method,label,color,ls,marker in methods:
  metrics=[np.array([[nr[method,s]['bins'][j]['mean_reward'] for j in range(7)] for s in range(20)]),np.array([[nr[method,s]['bins'][j]['mean_inventory_penalty'] for j in range(7)] for s in range(20)]),np.array([[lr[method,s]['bins'][j]['mean_loss'] for j in range(7)] for s in range(20)])]
  for ax,v in zip(axes,metrics):
   mean=v.mean(0);lo,hi=np.percentile(v[ix].mean(1),[2.5,97.5],axis=0);ax.plot(range(7),mean,label=label,color=color,ls=ls,marker=marker,ms=4);ax.fill_between(range(7),lo,hi,color=color,alpha=.08)
 for ax,title in zip(axes,['Reward (higher is better)','Inventory penalty (lower is better)','Decision loss (lower is better)']):
  ax.set(title=title,xticks=range(7),xticklabels=labels,xlabel='Decisions since switch');ax.tick_params(axis='x',rotation=40);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)
 axes[0].set_ylabel('Mean per decision');handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=4,frameon=False);fig.tight_layout(rect=(0,0,1,.90));a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output);plt.close(fig);print('Verified all 80 matching checkpoints and mean returns')
if __name__=='__main__':main()
