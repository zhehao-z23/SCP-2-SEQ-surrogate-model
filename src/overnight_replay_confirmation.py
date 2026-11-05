"""Fresh CPU replay of new confirmation checkpoints on length-spanning eval rows.

Prior checkpoints retain their original replay receipts; this checks only new fits.
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',required=True)
    ap.add_argument('--project',required=True)
    a=ap.parse_args(); root=Path(a.root); project=Path(a.project)
    sys.path.insert(0,str(project/'package/src'))
    from seq2scp_training.models import SequenceModel
    from pooling_model import PoolingModel
    torch.set_num_threads(2)
    data=pd.read_parquet(root/'data/samples.parquet').set_index('sample_id',drop=False)
    position=pd.Series(np.arange(len(data)),index=data.index)
    ix=pd.read_parquet(root/'cache/index.parquet')
    lookup=pd.Series(np.arange(len(ix)),index=ix.sequence_sha256)
    flat=np.load(root/'cache/flat.npy',mmap_mode='r')
    mean=np.load(root/'cache/mean.npy'); charge=np.load(root/'data/composition_charge.npy')
    fits=pd.read_csv(root/'results/all_fits.csv')
    checks=[]
    for row in fits.loc[~fits.reused].itertuples():
        out=Path(row.artifact_path)
        ck=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
        roles=json.loads((out/'roles.json').read_text()); ids=roles['evaluation_ids']
        meta=data.loc[ids]; order=np.argsort(meta.length.to_numpy(),kind='stable')
        take=order[np.linspace(0,len(order)-1,16).astype(int)]
        raw=charge[position.loc[meta.iloc[take].index].to_numpy()]
        flags=np.isnan(raw).astype('float32')
        if row.family=='composition_charge_deep':
            cfg=ck['config']
            ex=np.concatenate([(np.where(np.isnan(raw),ck['median'],raw)-ck['mean'])/ck['std'],flags],1)
            model=SequenceModel(input_dim=cfg['input_dim'],bins=63,kind='vector',pool='mean',hidden=cfg['hidden'],dropout=cfg['dropout']).eval()
            model.load_state_dict(ck['state_dict'])
            with torch.inference_mode(): pred=model(torch.tensor(ex))['logits'].softmax(1).numpy()
        else:
            ex=np.concatenate([(np.where(np.isnan(raw),ck['charge_median'],raw)-ck['charge_mean'])/ck['charge_std'],flags],1)
            model=PoolingModel(row.family).eval();model.load_state_dict(ck['state_dict']);pred=[]
            for n,k in enumerate(take):
                j=int(lookup[meta.iloc[k].sequence_sha256])
                if row.family=='mean_pre': x=mean[j][None];mask=None
                else:
                    z=ix.iloc[j];x=np.array(flat[int(z.offset):int(z.offset+z.length)],dtype='float32')[None]
                    mask=torch.ones(1,x.shape[1],dtype=torch.bool)
                x=(x-ck['embedding_mean'])/ck['embedding_std']
                with torch.inference_mode():pred.append(model(torch.tensor(x),torch.tensor(ex[n:n+1]),mask).softmax(1).numpy()[0])
            pred=np.asarray(pred)
        saved=np.load(out/'evaluation_prediction.npy')[take]
        error=float(np.max(abs(pred-saved)))
        assert np.allclose(pred,saved,rtol=1e-4,atol=2e-6),(row.run_id,error)
        checks.append(dict(run_id=row.run_id,family=row.family,fold=int(row.fold),seed=int(row.seed),rows=len(take),length_min=int(meta.iloc[take].length.min()),length_max=int(meta.iloc[take].length.max()),checkpoint_sha256=hashlib.sha256((out/'best.pt').read_bytes()).hexdigest(),CPU_fresh_reload_max_absolute_error=error,passed=True))
    report=dict(status='complete' if len(checks)==20 else 'partial',expected_new=20,replayed_new=len(checks),rows_per_checkpoint=16,max_absolute_error=max(x['CPU_fresh_reload_max_absolute_error'] for x in checks),checks=checks,scope='New confirmation checkpoints only; ten reused configurations retain prior replay receipts. No training or selection.')
    (root/'results/checkpoint_replay.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='checks'}))

if __name__=='__main__':main()
