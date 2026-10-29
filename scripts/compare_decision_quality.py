"""Compare quote quality using verified cached evaluation state-action counts."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
PAIRS=[('rl2_hypernet','rl2_concat','RL²: hypernet vs concat.'),('varibad_hypernet','varibad_concat','VariBAD: hypernet vs concat.'),('rl2_concat','varibad_concat','Concatenation: RL² vs VariBAD'),('rl2_hypernet','varibad_hypernet','Hypernetwork: RL² vs VariBAD')]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--table',type=Path);a=p.parse_args();ix=np.random.default_rng(0).integers(20,size=(10000,20));out={'description':'Positive reduction favours the first named configuration; unadjusted paired seed bootstrap.','comparisons':[],'cache_sha256':{}}
 def estimate(v):return {'mean':float(np.mean(v)),'ci95':np.percentile(np.array(v)[ix].mean(1),[2.5,97.5]).tolist()}
 for first,second,label in PAIRS:
  own=[];matched=[];overlap=[]
  for seed in range(20):
   path1=a.cache/f'{first}_{seed}.npz';path2=a.cache/f'{second}_{seed}.npz';x=np.load(path1);y=np.load(path2);rx=json.loads(str(x['row']));ry=json.loads(str(y['row']));assert rx['seed']==ry['seed']==seed
   for path in [path1,path2]:out['cache_sha256'][path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
   L=x['loss'];np.testing.assert_allclose(L,y['loss'],atol=0,rtol=0)
   cx,cy=x['counts'],y['counts'];nx,ny=cx.sum(-1),cy.sum(-1);w=np.minimum(nx,ny)
   lx=(cx*L).sum(-1)/np.maximum(nx,1);ly=(cy*L).sum(-1)/np.maximum(ny,1)
   own.append((ry['loss']-rx['loss'])/128);matched.append(float(((ly-lx)*w).sum()/w.sum()));overlap.append(float(w.sum()/cx.sum()))
  out['comparisons'].append({'first':first,'second':second,'label':label,'all_decisions':estimate(own),'comparable_situations':estimate(matched),'overlap':estimate(overlap)})
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['comparisons'],indent=2))
 if a.table:
  lines=[r'\begin{tabular}{llrr}',r'\toprule',r'Comparison & Decisions & Reduction & $95\%$ interval \\',r'\midrule']
  labels=[r'RL\textsuperscript{2}: hypernet vs concat.','VariBAD: hypernet vs concat.',r'Concat.: RL\textsuperscript{2} vs VariBAD',r'Hypernet: RL\textsuperscript{2} vs VariBAD']
  for label,c in zip(labels,out['comparisons']):
   for key,kind in [('all_decisions','All recorded'),('comparable_situations','Comparable')]:
    e=c[key];lines.append(label+' & '+kind+f" & ${e['mean']:.4f}$ & $[{e['ci95'][0]:.4f}, {e['ci95'][1]:.4f}]$"+r' \\')
  lines +=[r'\bottomrule',r'\end{tabular}'];a.table.parent.mkdir(parents=True,exist_ok=True);a.table.write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
