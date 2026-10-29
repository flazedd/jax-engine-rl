"""Render the verified frozen-policy decision-loss summary; no agent training."""
import argparse,json
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--tables',type=Path,required=True);p.add_argument('--figures',type=Path,required=True);a=p.parse_args();d=json.loads(a.input.read_text());a.tables.mkdir(parents=True,exist_ok=True);a.figures.mkdir(parents=True,exist_ok=True)
 def estimate(e,digits=2):return f"${e['mean']:+.{digits}f}$ & $[{e['ci95'][0]:+.{digits}f}, {e['ci95'][1]:+.{digits}f}]$"
 lines=[r'\begin{tabular}{llrr}',r'\toprule',r'Method & Component & Difference & $95\%$ interval \\',r'\midrule']
 for family,label in [('rl2',r'RL\textsuperscript{2}'),('varibad','VariBAD')]:
  for metric,name in [('return','Return'),('gain','Trading gains'),('penalty','Inventory penalty')]:lines.append(label+' & '+name+' & '+estimate(d['comparisons'][family][metric])+r' \\')
 lines += [r'\bottomrule',r'\end{tabular}'];(a.tables/'return_decomposition.tex').write_text('\n'.join(lines)+'\n')
 lines=[r'\begin{tabular}{llrr}',r'\toprule',r'Method & Decision loss & Difference & $95\%$ interval \\',r'\midrule']
 for family,label in [('rl2',r'RL\textsuperscript{2}'),('varibad','VariBAD')]:
  c=d['comparisons'][family];own={k:(v/128 if k=='mean' else [x/128 for x in v]) for k,v in c['loss'].items()}
  lines += [label+' & Own trajectories & '+estimate(own,4)+r' \\',label+' & Matched states & '+estimate(c['matched_loss_per_step'],4)+r' \\']
 lines += [r'\bottomrule',r'\end{tabular}'];(a.tables/'decision_loss.tex').write_text('\n'.join(lines)+'\n')
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':12,'pdf.fonttype':42})
 fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
 for ax,family,label in zip(axes,['rl2','varibad'],['RL²','VariBAD']):
  bins=d['comparisons'][family]['after_switch_loss'];v=np.array([b['difference']['mean'] for b in bins]);lo=np.array([b['difference']['ci95'][0] for b in bins]);hi=np.array([b['difference']['ci95'][1] for b in bins]);ax.errorbar(range(7),v,yerr=[v-lo,hi-v],fmt='o',color='black',capsize=4);ax.axhline(0,color='grey',lw=.8);ax.set(xticks=range(7),xticklabels=['0','1–4','5–9','10–19','20–39','40–79','80–126'],title=label,xlabel='Decisions since switch');ax.tick_params(axis='x',rotation=35);ax.spines[['top','right']].set_visible(False)
 axes[0].set_ylabel('Decision loss difference per decision\n(hypernetwork minus concatenation)');fig.tight_layout();fig.savefig(a.figures/'decision_loss_after_switch.pdf');plt.close(fig)
 print(json.dumps(d['comparisons'],indent=2))
if __name__=='__main__':main()
