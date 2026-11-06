"""Two fixed nonlinear shape readouts, leaving the accepted PMF model unchanged."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import sys
import time


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def load(p):return json.loads(Path(p).read_bytes())


def put(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--plan-sha256',required=True);ap.add_argument('--code-sha256',required=True);a=ap.parse_args()
    assert __debug__ and sha(a.plan)==a.plan_sha256 and sha(__file__)==a.code_sha256
    plan=load(a.plan);assert sorted(os.sched_getaffinity(0))==plan['resource']['CPU']
    assert os.environ['CUDA_VISIBLE_DEVICES']==plan['resource']['GPU']
    import numpy as np
    import pandas as pd
    import torch
    from torch import nn
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.mem_get_info()[0]>=4*(1<<30)
    start=time.monotonic();out=a.root/'nonlinear_probe_v1';assert not out.exists();out.mkdir()
    put(out/'started.json',{'PID':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
        'plan_sha256':sha(a.plan),'code_sha256':sha(__file__),'time_unix':time.time()})
    froot=a.root/'features_v1';lroot=a.root/'linear_probes_v1'
    assert sha(lroot/'summary.json')==plan['trigger_summary_sha256'];ls=load(lroot/'summary.json');fs=load(froot/'summary.json')
    assert sha(froot/'summary.json')==ls['feature_summary_sha256'] and fs['status']=='complete'
    assert sha(froot/'private_meta.parquet')==fs['private_meta_sha256'];meta=pd.read_parquet(froot/'private_meta.parquet')
    bridge=Path(os.environ['SCP_DATA_ROOT'])/'bridge_20261106';guard=bridge/'roles'
    assert sha(guard/'code/bridge_roles_guard.py')=='5401723f38305496ab11ab71d07825b2f1fe56476d753593d7929ed12a693cd3'
    sys.path.insert(0,str(guard/'code'));from bridge_roles_guard import check
    req={'stage':'development','sequence_sha256':sorted(meta.sequence_sha256.unique()),'aliases':[]}
    put(out/'role_preflight.json',[check(guard/'registry_v1/guard_contract.json','71359d1a06e18ab5dec65e825c82a07c3d7e0401c35266e8645957b943bb903c',{'operation':op,**req}) for op in ['read_target','train','stop','model_select','model_eval']])
    assert sha(lroot/'private_predictions.parquet')==ls['private_prediction_sha256']
    old=pd.read_parquet(lroot/'private_predictions.parquet');old=old.loc[old.method=='PMF_uniform_Bowley'].set_index('sample_id').loc[meta.sample_id].reset_index()
    assert np.array_equal(old.sequence_sha256,meta.sequence_sha256) and np.array_equal(old.group_id,meta.group_id)
    raw=old.raw_Bowley.to_numpy(float);binned=old.true63_uniform_Bowley.to_numpy(float);eligible=old.Bowley_eligible.to_numpy(bool);assert eligible.all()
    parts={}
    for part in ['train','stop','evaluation']:
        fp=froot/(part+'_indices.npy');assert sha(fp)==fs['outputs'][part+'_indices']['sha256'];parts[part]=np.load(fp)
    tr=parts['train'];va=parts['stop'];assert np.array_equal(old.partition.to_numpy()[va],np.full(len(va),'stop'))
    for left,right in [('train','stop'),('train','evaluation'),('stop','evaluation')]:
        assert not set(meta.iloc[parts[left]].group_id)&set(meta.iloc[parts[right]].group_id)
        assert not set(meta.iloc[parts[left]].sequence_sha256)&set(meta.iloc[parts[right]].sequence_sha256)
    hp=froot/'hidden256.npy';assert sha(hp)==fs['outputs']['hidden256']['sha256'];hidden=np.load(hp)
    cfg=next(x for x in ls['ridge_selection'] if x['feature']=='hidden256');npz=lroot/'hidden256_private_fit.npz'
    assert sha(npz)==cfg['coefficient_archive_sha256'];scaler=np.load(npz)
    X=(hidden.astype(float)-scaler['feature_mean'])/scaler['feature_std'];X[:,~scaler['active']]=0
    weights=1/meta.iloc[tr].sequence_sha256.map(meta.iloc[tr].sequence_sha256.value_counts()).to_numpy(float);weights/=weights.mean()
    ym=float(np.average(raw[tr],weights=weights));ys=float(np.sqrt(np.average((raw[tr]-ym)**2,weights=weights)));assert ys>1e-8
    xg=torch.tensor(X,dtype=torch.float32,device='cuda');wg=torch.tensor(weights,dtype=torch.float32,device='cuda')
    # Reduce row values exactly through teacher sequence then homology group.
    def group_mean(values,ids):
        m=meta.iloc[ids][['sequence_sha256','group_id']].copy();m['v']=values
        return float(m.groupby(['group_id','sequence_sha256']).v.mean().groupby('group_id').mean().mean())
    def predict(model,ids):
        model.eval();arr=[]
        with torch.inference_mode():
            for pos in range(0,len(ids),1024):arr.append((ym+ys*model(xg[ids[pos:pos+1024]]).squeeze(1)).cpu().numpy())
        return np.concatenate(arr)
    def digest_state(state):
        h=hashlib.sha256()
        for name,v in state.items():h.update(name.encode());h.update(v.detach().cpu().numpy().tobytes())
        return h.hexdigest()
    def rng_digest():
        return hashlib.sha256(torch.get_rng_state().numpy().tobytes()+torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()
    fit_results=[];models={};all_curves={}
    for arm,target in [('raw',raw),('binned',binned)]:
        random.seed(plan['seed']);np.random.seed(plan['seed']);torch.manual_seed(plan['seed']);torch.cuda.manual_seed_all(plan['seed'])
        model=nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,1)).cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        initial_digest=digest_state(model.state_dict());initial_rng=rng_digest();yg=torch.tensor((target[tr]-ym)/ys,dtype=torch.float32,device='cuda')
        pred=predict(model,va);best=group_mean((np.clip(pred,-1,1)-raw[va])**2,va);best_epoch=0
        state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};initial_stop=best;curve=[];fit_start=time.monotonic()
        for epoch in range(1,201):
            lr=next(x['value'] for x in plan['LR'] if x['first']<=epoch<=x['last'])
            for g in opt.param_groups:g['lr']=lr
            model.train();order=np.random.permutation(len(tr));total=0.;norms=[];bh=hashlib.sha256()
            for pos in range(0,len(tr),256):
                loc=order[pos:pos+256];rows=tr[loc];bh.update(rows.astype('<i8').tobytes())
                opt.zero_grad(set_to_none=True);estimate=model(xg[rows]).squeeze(1)
                loss=(((estimate-yg[loc])**2)*wg[loc]).mean();assert torch.isfinite(loss);loss.backward()
                norms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.)));opt.step();total+=float(loss.detach())*len(loc)
            pred=predict(model,va);score=group_mean((np.clip(pred,-1,1)-raw[va])**2,va)
            if score<best-1e-7:best=score;best_epoch=epoch;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            curve.append({'epoch':epoch,'lr':lr,'online_normalized_target_MSE':total/len(tr),'stop_raw_group_MSE':score,
                'batch_sha256':bh.hexdigest(),'end_RNG_sha256':rng_digest(),'clip_fraction':float(np.mean(np.asarray(norms)>1)),
                'gradient_norm_median':float(np.median(norms)),'elapsed_seconds':time.monotonic()-fit_start})
            if epoch%50==0:print(json.dumps({'arm':arm,'epoch':epoch,'stop_raw_MSE':score,'best_epoch':best_epoch}),flush=True)
            assert time.monotonic()-fit_start<plan['resource']['fit_cap_seconds_each']
        model.load_state_dict(state);torch.save({'state_dict':state,'feature_mean':scaler['feature_mean'],'feature_std':scaler['feature_std'],
            'active':scaler['active'],'target_mean':ym,'target_std':ys,'plan_sha256':sha(a.plan),'arm':arm},out/(arm+'_best.pt'))
        put(out/(arm+'_curve.json'),curve);models[arm]=model;all_curves[arm]=curve
        fit_results.append({'arm':arm,'epochs':200,'selected_epoch':best_epoch,'stop_raw_MSE':best,'initial_stop_raw_MSE':initial_stop,
            'initial_state_sha256':initial_digest,'initial_RNG_sha256':initial_rng,'fit_seconds':time.monotonic()-fit_start,
            'parameters':sum(p.numel() for p in model.parameters()),'checkpoint_sha256':sha(out/(arm+'_best.pt')),
            'curve_sha256':sha(out/(arm+'_curve.json'))})
    assert fit_results[0]['initial_state_sha256']==fit_results[1]['initial_state_sha256'] and fit_results[0]['initial_RNG_sha256']==fit_results[1]['initial_RNG_sha256']
    assert all(x['batch_sha256']==y['batch_sha256'] and x['end_RNG_sha256']==y['end_RNG_sha256'] for x,y in zip(all_curves['raw'],all_curves['binned']))
    helper=bridge/'diagnostic_supplements/code/bridge_shape_model_comparison.py';assert sha(helper)=='4be007ff1b869a4a71b800713f3e833f79b04770f7f27dd66aeb22507da16ce6'
    sys.path.insert(0,str(helper.parent));from bridge_shape_model_comparison import macro,confusion,rank_one_percent
    # All raw IQR are >=1 in the bound representation audit; truth is precisely that existing absolute rule.
    neg=raw<=-.3;pos=raw>=.3;strong=neg|pos
    assert {p:int(strong[ids].sum()) for p,ids in parts.items()}=={'train':49,'stop':13,'evaluation':16}
    results=[];private=[]
    for arm,model in models.items():
        bundle=torch.load(out/(arm+'_best.pt'),map_location='cpu',weights_only=False);replayed=nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,1)).cuda()
        replayed.load_state_dict(bundle['state_dict']);diff=0.
        for part,ids in parts.items():
            score=predict(model,ids);again=predict(replayed,ids);diff=max(diff,float(abs(score-again).max()));assert diff==0
            clipped=np.clip(score,-1,1);m=meta.iloc[ids].reset_index(drop=True);f=m[['sample_id','sequence_sha256','group_id']].copy()
            f['predicted_Bowley']=clipped;f['score_before_clip']=score;f['Bowley_eligible']=eligible[ids];f['raw_Bowley']=raw[ids]
            f['Bowley_AE']=abs(clipped-raw[ids]);f['Bowley_MSE']=(clipped-raw[ids])**2
            ranking,_=rank_one_percent(f.assign(predicted_Bowley=score),neg[ids],pos[ids])
            results.append({'arm':arm,'partition':part,'overall':macro(f,['Bowley_AE','Bowley_MSE']),
                'strong':{name:macro(f.loc[mask[ids]],['Bowley_AE','Bowley_MSE']) for name,mask in [('union',strong),('negative',neg),('positive',pos)]},
                'negative':confusion(m,neg[ids],eligible[ids]&(clipped<=-.3),eligible[ids]),'positive':confusion(m,pos[ids],eligible[ids]&(clipped>=.3),eligible[ids]),
                'ranking':ranking,'clipped_rows':int((abs(score)>1).sum()),'distinct_scores':int(np.unique(score).size)})
            private.append(f.assign(arm=arm,partition=part));np.save(out/(arm+'_'+part+'_prediction.npy'),score)
        next(x for x in fit_results if x['arm']==arm)['checkpoint_prediction_replay_maxdiff']=diff
    base={(x['method'],x['partition']):x for x in ls['results']};by={(x['arm'],x['partition']):x for x in results};decisions=[]
    for arm in ['raw','binned']:
        checks=[]
        for part in ['stop','evaluation']:
            b=base[('PMF_uniform_Bowley',part)];c=by[(arm,part)]
            btp=sum(b['ranking'][s]['true_positive_T'] for s in ['negative','positive']);ctp=sum(c['ranking'][s]['true_positive_T'] for s in ['negative','positive'])
            checks.append({'partition':part,'rare_AE_reduction10pct':c['strong']['union']['Bowley_AE']['group_macro']<=.9*b['strong']['union']['Bowley_AE']['group_macro'],
                'overall_MSE_within5pct':c['overall']['Bowley_MSE']['group_macro']<=1.05*b['overall']['Bowley_MSE']['group_macro'],
                'ranking_not_lower':ctp>=btp,'TP_delta':ctp-btp})
        passes=arm=='raw' and all(c['rare_AE_reduction10pct'] and c['overall_MSE_within5pct'] and c['ranking_not_lower'] for c in checks) and any(c['TP_delta']>=2 for c in checks)
        decisions.append({'arm':arm,'checks':checks,'promising_for_strong_control_comparison':bool(passes)})
    pd.concat(private,ignore_index=True).to_parquet(out/'private_predictions.parquet',index=False)
    assert sha(froot/'predicted_PMF.npy')==fs['outputs']['predicted_PMF']['sha256']
    summary={'status':'complete','schema':'bounded_nonlinear_shape_probe_result.v1','plan_sha256':sha(a.plan),'code_sha256':sha(__file__),
        'linear_trigger_sha256':sha(lroot/'summary.json'),'feature_summary_sha256':sha(froot/'summary.json'),'fits':fit_results,'results':results,'decisions':decisions,
        'paired_initialization_batch_and_dropout_RNG_checks':200,'PMF_unchanged':True,'new_backbone_training':False,
        'private_predictions_sha256':sha(out/'private_predictions.parquet'),'process_seconds':time.monotonic()-start,
        'max_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'peak_GPU_bytes':torch.cuda.max_memory_allocated(),
        'scope':'Two small auxiliary heads on a frozen, previously stop-selected development backbone; scalar outputs are not a self-consistent new PMF or evidence of a physical causal mechanism.'}
    put(out/'summary.json',summary);print(json.dumps({'status':'complete','summary_sha256':sha(out/'summary.json'),'fits':fit_results,'decisions':decisions}),flush=True)


if __name__=='__main__':main()
