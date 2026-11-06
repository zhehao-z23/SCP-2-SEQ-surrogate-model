"""Plot two completed loss-probe logs without comparing unlike online losses."""
from pathlib import Path
import argparse,hashlib,json,sys

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--logs',type=Path,required=True);p.add_argument('--manifest-sha256',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--plot-packages',type=Path);a=p.parse_args()
 if a.plot_packages:sys.path.insert(0,str(a.plot_packages))
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import numpy as np
 assert sha(a.logs/'mirror_manifest.json')==a.manifest_sha256
 m=json.loads((a.logs/'mirror_manifest.json').read_bytes())
 for n,r in m['files'].items():assert sha(a.logs/n)==r['sha256']
 arms=['H2','CE_quarter'];labels={'H2':'H2 control','CE_quarter':'CE/4'};colors={'H2':'#2c6c9c','CE_quarter':'#d7742b'}
 curves={x:json.loads((a.logs/x/'curve.json').read_bytes()) for x in arms};receipts={x:json.loads((a.logs/x/'receipt.json').read_bytes()) for x in arms}
 assert all(len(curves[x])==134 and receipts[x]['epochs']==134 for x in arms)
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(3,2,figsize=(15,13),dpi=140)
 for ax,x in zip(axes[0],arms):
  c=curves[x];ax.plot([v['epoch'] for v in c],[v['online_train_loss'] for v in c],color=colors[x]);ax.set_title(labels[x]+' online training loss (separate scale)');ax.set_ylabel('Hellinger squared' if x=='H2' else 'Cross-entropy / 4')
 for x in arms:
  c=curves[x];e=np.array([v['epoch'] for v in c]);h=np.array([v['stop_H'] for v in c]);best=receipts[x]['best_epoch']
  axes[1,0].plot(np.r_[0,e],np.r_[receipts[x]['initial_stop_H'],h],label=labels[x],color=colors[x])
  mask=e>=60;axes[1,1].plot(e[mask],h[mask],label=labels[x],color=colors[x]);axes[1,1].plot(best,receipts[x]['best_stop_H'],'o',color=colors[x],label=f'{labels[x]} selected epoch {best}')
  axes[2,0].plot(e,[v['stop_CE'] for v in c],label=labels[x],color=colors[x])
  axes[2,1].plot(e,[v['stop_CDF_MSE'] for v in c],label=labels[x],color=colors[x])
 axes[1,0].set_title('Same stop group-macro H: full scale including epoch 0');axes[1,0].set_ylabel('Hellinger');axes[1,0].legend()
 axes[1,1].set_title('Same stop H: explicit epoch 60-134 zoom');axes[1,1].set_ylabel('Hellinger');axes[1,1].legend(fontsize=8)
 axes[2,0].set_title('Same unscaled stop cross-entropy');axes[2,0].set_ylabel('Cross-entropy');axes[2,0].legend()
 axes[2,1].set_title('Same stop cutpoint CDF MSE (log scale)');axes[2,1].set_ylabel('CDF MSE');axes[2,1].set_yscale('log');axes[2,1].legend()
 for ax in axes.flat:
  ax.axvline(84,color='#657382',ls=':',lw=1);ax.axvline(102,color='#657382',ls=':',lw=1);ax.set_xlabel('Completed epoch');ax.grid(alpha=.18)
 fig.suptitle('Fixed-path loss probe: 134 epochs, one fold / seed\nH2 exact control replay; CE/4 does not pass the frozen promotion rule',fontsize=15)
 fig.text(.035,.022,'Dotted lines: LR changes before epochs 84 and 102 (3e-4 / 1e-4 / 3e-5). Each arm selects its own best by the same stop-H rule.\nOnline losses have different definitions and must not be subtracted. CE best is at the budget endpoint; convergence is not established.\nMean clip fraction = 0 and stop-CE probability-floor count = 0 in both arms. No epochwise fixed-state train-evaluation curve is invented.',fontsize=10)
 fig.tight_layout(rect=(0,.08,1,.94));a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output);plt.close(fig)
 receipt={'status':'rendered_from_verified_saved_logs','plot_sha256':sha(a.output),'source_sha256':sha(__file__),'mirror_manifest_sha256':a.manifest_sha256,'epochs_per_arm':134,'selected_epochs':{x:receipts[x]['best_epoch'] for x in arms},'no_fits_or_inference':True,'private_only':True}
 a.output.with_suffix('.receipt.json').write_bytes((json.dumps(receipt,indent=2)+'\n').encode());print(json.dumps(receipt))

if __name__=='__main__':main()
