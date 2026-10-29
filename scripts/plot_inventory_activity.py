"""Regenerate inventory and fill summaries from saved evaluation statistics."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

METHODS=[('rl2_concat','RL² concatenation','#275f8a','--'),
         ('rl2_hypernet','RL² hypernetwork','#275f8a','-'),
         ('varibad_concat','VariBAD concatenation','#178c7b','--'),
         ('varibad_hypernet','VariBAD hypernetwork','#178c7b','-')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--figure',type=Path,required=True)
    p.add_argument('--table',type=Path,required=True)
    a=p.parse_args();data=json.loads(a.input.read_text())
    plt.rcParams.update({'font.size':13,'pdf.fonttype':42,'font.family':'DejaVu Sans'})
    fig,(left,right)=plt.subplots(1,2,figsize=(10.5,4.0),gridspec_kw={'width_ratios':[1.25,1]})
    x=np.arange(6)
    for i,(method,label,color,style) in enumerate(METHODS):
        d=data['methods'][method]['absolute_inventory_distribution']
        left.plot(x,100*np.asarray(d['mean']),label=label,color=color,linestyle=style,
                  marker='o' if style=='-' else 's',markersize=4)
        left.fill_between(x,100*np.asarray(d['ci95'][0]),100*np.asarray(d['ci95'][1]),color=color,alpha=.10)
        f=data['methods'][method]['total_fills'];v=f['mean'];lo,hi=f['ci95']
        right.errorbar(v,3-i,xerr=[[v-lo],[hi-v]],color=color,fmt='o' if style=='-' else 's',
                       markerfacecolor=color if style=='-' else 'white',capsize=4,markersize=6)
    left.set(xticks=x,xlabel='Absolute inventory after fills',ylabel='Share of decisions (%)',
             title='Inventory exposure',ylim=(0,None))
    left.legend(handles=[Line2D([0],[0],color='#275f8a',label='RL²'),
                         Line2D([0],[0],color='#178c7b',label='VariBAD')],
                frameon=False,fontsize=12,loc='upper right')
    right.set(yticks=[3,2,1,0],yticklabels=[m[1] for m in METHODS],xlabel='Accepted fills per episode',
              title='Trading activity',ylim=(-.6,3.6))
    for ax in [left,right]:ax.spines[['top','right']].set_visible(False)
    right.spines['left'].set_visible(False);right.tick_params(axis='y',length=0)
    fig.tight_layout(w_pad=2);a.figure.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.figure,bbox_inches='tight');plt.close(fig)
    rows=[]
    for method,label,_,_ in METHODS:
        d=data['methods'][method]
        vals=[f"${d['mean_abs_inventory']['mean']:.3f}$",f"${100*d['large_inventory_fraction']['mean']:.1f}\\%$",
              *[f"${d[m]['mean']:.2f}$" for m in ['bid_fills','ask_fills','total_fills']]]
        label=label.replace('RL²',r'RL\textsuperscript{2}')
        rows.append(label+' & '+' & '.join(vals)+r' \\')
    text=r'''\begin{tabular}{lrrrrr}
\toprule
Agent & Mean $|q|$ & $P(|q|\geq3)$ & Bid fills & Ask fills & Total fills \\
\midrule
'''+ '\n'.join(rows)+ '\n\\bottomrule\n\\end{tabular}\n'
    text += r'\par\medskip'+'\n'+r'''\begin{tabular}{lrrrr}
\toprule
Comparison & $\Delta$ mean $|q|$ & $95\%$ interval & $\Delta$ total fills & $95\%$ interval \\
\midrule
'''
    labels=[('rl2',r'RL\textsuperscript{2}: hypernet $-$ concat.'),
            ('varibad',r'VariBAD: hypernet $-$ concat.'),
            ('concat_method',r'Concat.: RL\textsuperscript{2} $-$ VariBAD'),
            ('hypernet_method',r'Hypernet: RL\textsuperscript{2} $-$ VariBAD')]
    for key,label in labels:
        d=data['comparisons'][key];v=d['mean_abs_inventory'];f=d['total_fills']
        text += (f"{label} & ${v['mean']:+.3f}$ & $[{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}]$"
                 f" & ${f['mean']:+.2f}$ & $[{f['ci95'][0]:+.2f},{f['ci95'][1]:+.2f}]$"+r' \\'+'\n')
    text += '\\bottomrule\n\\end{tabular}\n'
    a.table.parent.mkdir(parents=True,exist_ok=True);a.table.write_text(text)

if __name__=='__main__':main()
