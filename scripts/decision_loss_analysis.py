"""Frozen-checkpoint return accounting and finite-horizon decision loss.
No training. Replays the published evaluation keys and verifies each return.
"""
from pathlib import Path
import json,hashlib
import jax
import numpy as np
from evaluation.action_distribution import collect_action_regime_rollouts
from evaluation.posterior_probe import load_experiment
from scripts.post_training_evaluation import METHODS,N_EPISODES,EPISODE_LENGTH
from scripts.natural_switch_evaluation import CELLS,BINS
from oracles.value_iteration import _transition_tables
from envs.market_making_v1 import MarketMakingV1
from utils.paths import analysis_dir,experiment_dir
ROOT=Path(__file__).resolve().parents[1]
OUT=analysis_dir()/'decision_loss';OUT.mkdir(exist_ok=True,parents=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def finite_values(env):
 P,R=_transition_tables(env);H=EPISODE_LENGTH;V=np.zeros((H+1,*R.shape[:2]));Q=np.zeros((H,*R.shape))
 np.testing.assert_allclose(P.sum(axis=(-1,-2)),1,atol=1e-12)
 for t in range(H-1,-1,-1):
  Q[t]=R+np.einsum('izajk,jk->iza',P,V[t+1]);V[t]=Q[t].max(-1)
 loss=V[:-1,...,None]-Q
 assert loss.min()>-1e-12
 np.testing.assert_allclose(loss.min(-1),0,atol=1e-12)
 return V,Q,loss

def interval(x):
 x=np.array(x);ix=np.random.default_rng(0).integers(len(x),size=(10000,len(x)))
 return {'mean':float(x.mean()),'ci95':np.percentile(x[ix].mean(1),[2.5,97.5]).tolist()}
def main():
 reference=json.loads((analysis_dir()/'m5r_post_training_evaluation.json').read_text())['methods'];rows=[];counts={};models={}
 for index,(method,exp) in enumerate(METHODS):
  if method not in CELLS:continue
  directory=experiment_dir(exp)
  for seed in range(20):
   ck=directory/f'checkpoint_seed_{seed}.pkl';dest=OUT/f'{method}_{seed}.npz';identity={'checkpoint':sha(ck),'config':sha(directory/'config.json'),'script':sha(__file__),'rollout':sha(ROOT/'evaluation/action_distribution.py')}
   if dest.exists():
    data=np.load(dest);row=json.loads(str(data['row']));assert row['identity']==identity;count=data['counts'];loss=data['loss'];V0=float(data['V0'])
   else:
    bundle=load_experiment(directory,seed);params=bundle.config['env']['params'];env=MarketMakingV1(**params);V,Q,loss=finite_values(env);V0=float(np.dot(V[0,env.inventory_max],env.initial_distribution))
    rollout=collect_action_regime_rollouts(bundle.env,bundle.agent,bundle.agent_state,N_EPISODES,EPISODE_LENGTH,jax.random.PRNGKey(1000000+10000*index+seed))
    r=rollout['reward'].astype(float);post=rollout['post_inventory'];q=rollout['inventory'];z=rollout['regime'];a=rollout['action'];pen=env.inventory_penalty*post.astype(float)**2;gain=r+pen;t=np.arange(EPISODE_LENGTH)[:,None];dl=loss[t,q+env.inventory_max,z,a]
    ret=float(r.sum(0).mean());expected=dict(zip(reference[method]['seeds'],reference[method]['per_seed_evaluation_return']))[seed];np.testing.assert_allclose(ret,expected,atol=1e-10,rtol=0)
    np.testing.assert_allclose(gain.sum(0).mean()-pen.sum(0).mean(),ret,atol=1e-12)
    # Trading gains must be sums of accepted one- or three-tick fills.
    np.testing.assert_allclose(gain,np.rint(gain),atol=2e-6)
    count=np.zeros((128,11,3,3),dtype=np.int64);np.add.at(count,(np.broadcast_to(t,a.shape),q+env.inventory_max,z,a),1)
    changed=np.zeros(z.shape,bool);changed[1:]=z[1:]!=z[:-1];latest=np.maximum.accumulate(np.where(changed,t,-1),axis=0);age=np.where(latest>=0,t-latest,-1)
    bins=[]
    for lo,hi in BINS:
     mask=(age>=lo)&(age<=hi);bins.append({'steps':int(mask.sum()),'mean_loss':float(dl[mask].mean()) if mask.any() else None})
    row={'method':method,'seed':seed,'identity':identity,'return':ret,'gain':float(gain.sum(0).mean()),'penalty':float(pen.sum(0).mean()),'loss':float(dl.sum(0).mean()),'mean_abs_inventory':float(np.abs(post).mean()),'bound_fraction':float((np.abs(post)==5).mean()),'bins':bins,'oracle_optimum':V0,'oracle_gap_minus_loss':V0-ret-float(dl.sum(0).mean())}
    np.savez_compressed(dest,row=json.dumps(row),counts=count,loss=loss,V0=V0)
   rows.append(row);counts[method,seed]=count;models[method,seed]=loss
   print(method,seed,'verified',flush=True)
 summary={'description':'Observed return accounting; full-information finite-horizon action loss. No retraining.','intervals':'10000 paired seed bootstrap, unadjusted exploratory intervals','rows':rows,'methods':{},'comparisons':{}}
 for method in CELLS:
  rr=[r for r in rows if r['method']==method];summary['methods'][method]={metric:interval([r[metric] for r in rr]) for metric in ['return','gain','penalty','loss','mean_abs_inventory','bound_fraction','oracle_gap_minus_loss']}
 for family in ['rl2','varibad']:
  c,h=family+'_concat',family+'_hypernet';cr=[r for r in rows if r['method']==c];hr=[r for r in rows if r['method']==h];comp={metric:interval([b[metric]-a[metric] for a,b in zip(cr,hr)]) for metric in ['return','gain','penalty','loss','mean_abs_inventory','bound_fraction']}
  matched=[];coverage=[]
  for seed in range(20):
   ca,cb=counts[c,seed],counts[h,seed];na,nb=ca.sum(-1),cb.sum(-1);w=np.minimum(na,nb);L=models[c,seed];np.testing.assert_allclose(L,models[h,seed],atol=0)
   la=(ca*L).sum(-1)/np.maximum(na,1);lb=(cb*L).sum(-1)/np.maximum(nb,1);matched.append(float(((lb-la)*w).sum()/w.sum()));coverage.append(float(w.sum()/(128*512)))
  comp['matched_loss_per_step']=interval(matched);comp['matched_overlap_fraction']=interval(coverage)
  comp['after_switch_loss']=[{'interval':list(bounds),'difference':interval([b['bins'][j]['mean_loss']-a['bins'][j]['mean_loss'] for a,b in zip(cr,hr)])} for j,bounds in enumerate(BINS)]
  summary['comparisons'][family]=comp
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(OUT/'summary.json',flush=True)
if __name__=='__main__':main()
