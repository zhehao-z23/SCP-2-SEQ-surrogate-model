"""Guarded new-area-label development adapter and bounded CPU interface smoke.

All paths and authority pins are supplied by a frozen configuration. This entry
does not authorize a final release, use N=1 as validation, or translate old bins.
Private sequences, labels, checkpoints and exposure membership stay at output.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def load(path):return json.loads(Path(path).read_text())
def put(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def verified(ref):
    p=Path(ref['path'])
    if sha(p)!=ref['sha256']:raise ValueError('changed source '+str(p))
    return p
def module(name,ref):
    p=verified(ref);spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def authority(cfg,operation,sequence_sha):
    guard=module('transition_roles_guard',cfg['guard_code'])
    return guard.check(verified(cfg['guard_contract']),cfg['guard_contract']['sha256'],
        {'operation':operation,'stage':'development','sequence_sha256':[sequence_sha],'aliases':[]})


def resource_preflight(cfg):
    verified(cfg['runtime_upgrade']);verified(cfg['resource_inventory'])
    inv=load(cfg['resource_inventory']['path'])
    verified(inv['controller']);verified(inv['release_manifest']);verified(inv['phase'])
    assert inv['status']=='passed' and inv['plan_cpu_slots']==64
    assert set(os.sched_getaffinity(0))<=set(cfg['cpu_affinity'])
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert all(os.environ.get(x)=='1' for x in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'])
    physical=lambda c:tuple(int((Path('/sys/devices/system/cpu')/f'cpu{c}'/'topology'/k).read_text()) for k in ['physical_package_id','core_id'])
    ours={physical(c) for c in cfg['cpu_affinity']}
    assert not ours & {physical(c) for c in inv['production_representatives']}
    n=0
    for d in Path('/proc').iterdir():
        if not d.name.isdigit():continue
        try:
            cg=(d/'cgroup').read_text()
            if '/atlas_full_dC_' in cg and '/case_' in cg:
                assert not ours & {physical(c) for c in os.sched_getaffinity(int(d.name))};n+=1
        except (OSError,ProcessLookupError):pass
    return {'worker_processes_checked':n,'worker_slots_separate':True,'OS_exclusive':False,'affinity':sorted(os.sched_getaffinity(0))}


def validate_label_contract(contract,expected_bins_sha,expected_protocol_sha):
    assert contract['measure']=='area' and contract['equation']=='npbe'
    assert contract['potential_unit']=='kBT/e' and contract['primary_array']=='corrected_area_p'
    assert contract['bin_source_sha256']==expected_bins_sha and contract['protocol_sha256']==expected_protocol_sha
    assert contract['sequence_view']=='prepared_observed' and contract['input_matches_prepared'] is True
    assert contract['role']=='development_only' and contract['qualification_status']=='accepted'
    assert contract['n_bins']==len(contract['internal_edges'])+1
    assert all(a<b for a,b in zip(contract['internal_edges'],contract['internal_edges'][1:]))
    return True


def final_release_gate(candidate):
    # A different, reviewed release issuer must certify complete component/profile
    # closure and all old train/stop/model-selection exposure before final use.
    if candidate.get('stage')=='final_release':
        raise PermissionError('this development adapter cannot grant final release; complete exposure/group/profile closure required')
    assert candidate.get('stage')=='development'
    return {'allowed':True,'development_only':True,'final_independent_initialization_allowed':False}


def prepare(cfg,out):
    import numpy as np
    import pandas as pd
    receipt_path=verified(cfg['accepted_receipt']);r=load(receipt_path)
    assert r['status']=='accepted' and r['numerical_qualified'] is True and r['role']=='development_only'
    assert r['required_axes']==r['self_executed_axes'] and len(r['required_axes'])==8
    assert r['accepted_operator_variant']=='base' and r['primary_array']=='corrected_area_p'
    sid=r['source_identity']['sequence_sha256']
    # Guard precedes any target/vector/checkpoint read. Identity-only prepared
    # metadata is hash bound and checked again before consumption.
    guards=[authority(cfg,op,sid) for op in ['read_target','train','predict']]
    put(out/'role_preflight.json',guards)
    for key in ['qualification','integrity_audit','independent_audit','development_role_guard']:
        verified(r[key])
    s=load(verified(r['sequence_and_structure']))
    sequence=s['structure']['prepared_observed_sequence']
    assert hashlib.sha256(sequence.encode()).hexdigest()==sid==s['structure']['prepared_observed_sequence_sha256']
    assert sequence==s['structure']['entity_sequence'] and len(sequence)==r['source_identity']['length']
    # This N=1 pilot is complete-coordinate T=U. General records must declare
    # exact prepared input and coverage; full-U predictions on partial-T targets
    # require a distinct task and are not silently admitted here.
    manifest=load(verified(r['label_representation']))
    assert manifest['primary_array']=='corrected_area_p' and manifest['potential_unit']=='kBT/e'
    assert manifest['spec']['inputs']==r['source_inputs']
    vector_ref=next(p for p in r['products'] if Path(p['path']).name=='vectors.npz')
    assert next(p for p in manifest['products'] if p['name']=='vectors.npz')['sha256']==vector_ref['sha256']
    bins=load(verified(r['source_inputs']['shared_bins']))
    protocol=load(verified(r['source_inputs']['protocol_configuration']))
    assert protocol['PB_physical_parameters']['equation']=='npbe'
    assert {i['concentration_M'] for i in protocol['PB_physical_parameters']['ions']}=={.1}
    assert s['PQR']['forcefield']=='PARSE' and s['PQR']['chemical_integrity_status']=='passed'
    internal=np.asarray(bins['internal_edges'],dtype=np.float64)
    with np.load(verified(vector_ref),allow_pickle=False) as z:
        y=z['corrected_area_p'].copy()
        assert y.shape==(len(internal)+1,) and np.isfinite(y).all() and (y>=0).all()
        assert abs(y.sum()-1)<1e-6
        assert np.allclose(np.sqrt(y),z['corrected_area_sqrt_p'],atol=1e-12)
    contract={'schema':'atlas_new_area_development_contract.v1','role':'development_only','qualification_status':'accepted','measure':'area','primary_array':'corrected_area_p','equation':'npbe','salt_M':.1,'potential_unit':'kBT/e','sequence_view':'prepared_observed','input_matches_prepared':True,'T_equals_U_for_this_pilot':True,'n_bins':len(y),'internal_edges':internal.tolist(),'bin_rule':'[-inf, internal_edges, +inf), edge ties enter right bin','bin_source_sha256':r['source_inputs']['shared_bins']['sha256'],'protocol_sha256':r['source_inputs']['protocol_configuration']['sha256'],'accepted_receipt_sha256':sha(receipt_path),'not_a_final_release':True}
    validate_label_contract(contract,cfg['expected_bin_source_sha256'],cfg['expected_protocol_sha256'])
    verified(cfg['charge_code']);verified(cfg['feature_schema_code'])
    sys.path.insert(0,str(Path(cfg['charge_code']['path']).parent.parent))
    from seq2scp_training import charge_features as charge,features
    row=charge.extract_features(sequence,sequence_view='prepared_observed',full_chain=True)
    cols=features.VIEWS['composition_charge'];values=np.asarray([np.nan if row[k] is None else row[k] for k in cols],np.float32)
    assert len(cols)==39 and row['sequence_sha256']==sid and not np.isinf(values).any()
    np.save(out/'target.private.npy',y[None].astype(np.float32),allow_pickle=False)
    np.save(out/'charge.private.npy',values[None],allow_pickle=False)
    put(out/'sequence.private.json',{'sequence':sequence,'sequence_sha256':sid,'length':len(sequence)})
    put(out/'label_contract.json',contract)
    matches=[]
    for cache in cfg['known_caches']:
        cr=load(verified(cache['receipt']));index=verified(cache['index'])
        rows=load(index) if index.suffix=='.json' else pd.read_parquet(index).to_dict('records')
        selected=[x for x in rows if x['sequence_sha256']==sid]
        assert len(selected)<=1
        matches.append({'receipt_sha256':sha(cache['receipt']['path']),'exact_sequence_match_count':len(selected),'index_sha256':sha(index)})
    # Existing caches are keyed by exact input sequence, never AF accession.
    # The local extraction step is explicit even when no existing row matches.
    put(out/'cache_lookup.json',matches)
    private={'accepted_receipt':cfg['accepted_receipt'],'sequence_sha256':sid,'sequence_and_structure':r['sequence_and_structure'],'target_product':vector_ref,'role_preflight_sha256':sha(out/'role_preflight.json'),'feature_columns':cols,'feature_code_sha256':cfg['charge_code']['sha256'],'files':{p.name:sha(p) for p in out.iterdir() if p.is_file()}}
    put(out/'prepared_private.json',private)
    return {'status':'complete','accepted_labels':1,'sequence_views_equal':True,'target_bins':len(y),'existing_cache_exact_hits':sum(x['exact_sequence_match_count'] for x in matches),'guard_before_target':True,'raw_phi_read':False,'PB_calls':0}


def embed(cfg,out):
    import numpy as np
    import torch
    prepared=out.parent/'prepared';binding=load(prepared/'prepared_private.json')
    for name,pin in binding['files'].items():assert sha(prepared/name)==pin
    row=load(prepared/'sequence.private.json');guard=authority(cfg,'predict',row['sequence_sha256'])
    verified(cfg['backend_code']);verified(cfg['encoder_manifest'])
    sys.path.insert(0,cfg['backend_src'])
    from seq2scp_training.hf_backend import LocalHFBackend
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    backend=LocalHFBackend(load(cfg['encoder_manifest']['path']),device='cpu')
    assert not backend.model.training and not any(p.requires_grad for p in backend.model.parameters())
    start=time.monotonic();z=backend.extract_batch([row['sequence']],[30],precision='fp32')[0]
    assert z['sequence_sha256']==row['sequence_sha256']
    x=z['layers'][30];assert x.shape==(row['length'],960) and np.isfinite(x).all()
    q=x.astype(np.float16);assert np.isfinite(q).all()
    np.save(out/'residues.private.npy',q,allow_pickle=False)
    receipt={'status':'complete','sequence_sha256':row['sequence_sha256'],'input_sha256':sha(prepared/'sequence.private.json'),'encoder_manifest_sha256':cfg['encoder_manifest']['sha256'],'backend_sha256':cfg['backend_code']['sha256'],'encoder':'esmc_300m','layer':30,'width':960,'compute':'FP32 CPU singleton','cache_dtype':'float16','no_truncation':True,'special_tokens_excluded':True,'encoder_frozen':True,'guard':guard,'flat_sha256':sha(out/'residues.private.npy'),'forward_seconds':time.monotonic()-start,'torch':torch.__version__,'numpy':np.__version__}
    put(out/'cache_receipt.private.json',receipt)
    return {'status':'complete','rows':1,'frozen_encoder':True,'width':960,'GPU_used':False,'forward_seconds':receipt['forward_seconds']}


def make_model(mode,bins):
    import torch
    from torch import nn
    from torch.nn import functional as F
    class Model(nn.Module):
        def __init__(self):
            super().__init__();self.first=nn.Linear(78 if mode=='charge_deep' else 1038,1024)
            self.head=nn.Sequential(nn.Dropout(.3),nn.Linear(1024,512),nn.GELU(),nn.Dropout(.3),nn.Linear(512,256),nn.GELU(),nn.Dropout(.3),nn.Linear(256,bins))
        def forward(self,x,extra,mask):
            if mode=='charge_deep':z=F.gelu(self.first(extra))
            elif mode=='mean_pre':z=F.gelu(self.first(torch.cat([x.mean(1),extra],1)))
            elif mode=='residue_mean':
                assert mask.any(1).all()
                z=F.gelu(F.linear(x,self.first.weight[:,:960])+F.linear(extra,self.first.weight[:,960:],self.first.bias)[:,None]);z=(z*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]
            else:raise ValueError('unknown architecture')
            return self.head(z)
    return Model()


def transfer_hidden(model,bundle,new_scalers,mode):
    """Rebase the first affine map onto fit-only new scalers; reset the output.

    Missing flags absorb the change of charge imputation median. This preserves
    old preactivations on a fixed raw input while both arms consume the same new
    normalized input. No old output, optimizer state or RNG is imported.
    """
    import numpy as np
    import torch
    for scales in [bundle,new_scalers]:
        for key in ['embedding_std','charge_std']:
            assert np.isfinite(scales[key]).all() and (np.asarray(scales[key])>0).all()
        for key in ['embedding_mean','charge_median','charge_mean']:
            assert np.isfinite(scales[key]).all()
    original={k:v.detach().clone() for k,v in model.state_dict().items()}
    old=bundle['state_dict'];state=model.state_dict()
    for k in state:
        if not k.startswith('head.7.'):
            assert k in old and state[k].shape==old[k].shape
            state[k]=old[k].detach().clone()
    W=old['first.weight'].detach().clone();b=old['first.bias'].detach().clone()
    offset=0 if mode=='charge_deep' else 960
    if offset:
        ratio=np.asarray(new_scalers['embedding_std'])/bundle['embedding_std']
        shift=(np.asarray(new_scalers['embedding_mean'])-bundle['embedding_mean'])/bundle['embedding_std']
        state['first.weight'][:,:960]=W[:,:960]*torch.tensor(ratio,dtype=W.dtype)
        b+=W[:,:960]@torch.tensor(shift,dtype=W.dtype)
    ratio=np.asarray(new_scalers['charge_std'])/bundle['charge_std']
    shift=(np.asarray(new_scalers['charge_mean'])-bundle['charge_mean'])/bundle['charge_std']
    missing=(bundle['charge_median']-np.asarray(new_scalers['charge_median']))/bundle['charge_std']
    state['first.weight'][:,offset:offset+39]=W[:,offset:offset+39]*torch.tensor(ratio,dtype=W.dtype)
    state['first.weight'][:,offset+39:offset+78]=W[:,offset+39:offset+78]+W[:,offset:offset+39]*torch.tensor(missing,dtype=W.dtype)
    state['first.bias']=b+W[:,offset:offset+39]@torch.tensor(shift,dtype=W.dtype)
    model.load_state_dict(state)
    assert all(torch.equal(model.state_dict()[k],original[k]) for k in ['head.7.weight','head.7.bias'])


def smoke(cfg,out):
    import numpy as np
    import pandas as pd
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    root=out.parent;prepared=root/'prepared';binding=load(prepared/'prepared_private.json')
    for name,pin in binding['files'].items():assert sha(prepared/name)==pin
    sid=binding['sequence_sha256'];guards=[authority(cfg,op,sid) for op in ['read_target','train','predict']]
    final_release_gate({'stage':'development'})
    contract=load(prepared/'label_contract.json');validate_label_contract(contract,cfg['expected_bin_source_sha256'],cfg['expected_protocol_sha256'])
    cr=load(root/'embedding/cache_receipt.private.json');assert cr['sequence_sha256']==sid and cr['encoder_manifest_sha256']==cfg['encoder_manifest']['sha256']
    assert sha(root/'embedding/residues.private.npy')==cr['flat_sha256']
    x=np.load(root/'embedding/residues.private.npy').astype(np.float32)[None]
    y=np.load(prepared/'target.private.npy');charge=np.load(prepared/'charge.private.npy')
    assert x.shape[2]==960 and y.shape==(1,contract['n_bins']) and charge.shape==(1,39)
    means=x.mean(1);med=np.nan_to_num(np.nanmedian(charge,axis=0));imputed=np.where(np.isnan(charge),med,charge)
    scalers={'embedding_mean':means.mean(0),'embedding_std':np.ones(960,np.float32),'charge_median':med,'charge_mean':imputed.mean(0),'charge_std':np.ones(39,np.float32)}
    # N=1 zero-variance fallback is an interface test, never a population scaler.
    xt=torch.tensor((x-scalers['embedding_mean'])/scalers['embedding_std']);extra=torch.tensor(np.concatenate([(imputed-scalers['charge_mean'])/scalers['charge_std'],np.isnan(charge).astype(np.float32)],1));mask=torch.ones(x.shape[:2],dtype=torch.bool);target=torch.tensor(y)
    checkpoint=verified(cfg['historical_checkpoint']);receipt=load(verified(cfg['historical_receipt']));roles=load(verified(cfg['historical_roles']))
    assert receipt['status']=='complete' and receipt['checkpoint_sha256']==sha(checkpoint)
    assert receipt['config']['mode']=='residue_mean' and receipt['config']['policy']=='plateau'
    # Historical checkpoint belongs solely to the previously released development
    # cohort. Preserve all three exposure sets; no old label or prediction IO.
    release=load(verified(cfg['historical_release']));meta_path=verified(cfg['historical_meta'])
    assert release['artifacts']['samples.parquet']==sha(meta_path)
    meta=pd.read_parquet(meta_path,columns=['sample_id','sequence_sha256','group_id'])
    lookup=meta.set_index('sample_id');assert lookup.index.is_unique
    exposures={k:lookup.loc[ids,['sequence_sha256','group_id']].drop_duplicates().to_dict('records') for k,ids in roles.items()}
    assert set(roles)=={'train_ids','stop_ids','evaluation_ids'}
    put(out/'historical_exposure.private.json',{'partitions':exposures,'all_historical_development_selected_or_reported':True,'existing_release_role_gate_not_superseded':True,'eligible_for_independent_initialization':False,'missing':'new-release full homology/profile closure and training+stopping+model-selection exclusion'})
    bundle=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert bundle['config']==receipt['config'] and bundle['config']['release_sha256']==cfg['historical_release']['sha256']
    results=[];initial_heads=[];replay=[];curves=[]
    for arm in ['from_scratch','historical_hidden_rebased']:
        torch.manual_seed(cfg['seed']);model=make_model('residue_mean',contract['n_bins'])
        if arm=='historical_hidden_rebased':transfer_hidden(model,bundle,scalers,'residue_mean')
        initial_heads.append({k:v.clone() for k,v in model.state_dict().items() if k.startswith('head.7.')})
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['smoke_lr'],weight_decay=1e-4)
        assert not optimizer.state
        def prediction():
            model.eval()
            with torch.inference_mode():return model(xt,extra,mask).softmax(1)
        def loss():
            logp=model(xt,extra,mask).log_softmax(1)
            return .5*((torch.exp(.5*logp)-target.sqrt())**2).sum(1).mean()
        before=prediction();torch.manual_seed(cfg['seed']+1);curve=[]
        for step in range(cfg['smoke_updates']):
            model.train();optimizer.zero_grad(set_to_none=True);v=loss();assert torch.isfinite(v);v.backward()
            grads=[p.grad for p in model.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads) and any(g.abs().max()>0 for g in grads)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();curve.append(float(v.detach()))
        after=prediction();assert torch.isfinite(after).all() and abs(float(after.sum())-1)<1e-6
        ck=out/(arm+'.private.pt');torch.save({'state_dict':model.state_dict(),'scalers':scalers,'label_contract':contract,'mode':'residue_mean','historical_final_eligible':False},ck)
        restored=make_model('residue_mean',contract['n_bins']).eval();restored.load_state_dict(torch.load(ck,map_location='cpu',weights_only=False)['state_dict'])
        with torch.inference_mode():diff=float((restored(xt,extra,mask).softmax(1)-after).abs().max())
        assert diff==0
        np.save(out/(arm+'.prediction.private.npy'),after.numpy(),allow_pickle=False)
        results.append({'arm':arm,'steps':len(curve),'finite_gradients':True,'probability_valid':True,'checkpoint_replay_max_difference':diff,'parameter_count':sum(p.numel() for p in model.parameters())})
        curves.append({'arm':arm,'training_H2':curve,'initial_H2':float(.5*((before.sqrt()-target.sqrt())**2).sum()),'final_H2':float(.5*((after.sqrt()-target.sqrt())**2).sum())})
    assert all(torch.equal(initial_heads[0][k],initial_heads[1][k]) for k in initial_heads[0])
    put(out/'loss_curves.private.json',curves);put(out/'guard_receipts.private.json',guards)
    return {'status':'complete','N':1,'arms':results,'same_new_output_initialization':True,'same_new_fit_only_scalers':True,'old_output_discarded':True,'old_optimizer_or_RNG_imported':False,'new_fit_updates_total':2*cfg['smoke_updates'],'stop_or_test_partition':False,'scientific_comparison_or_generalization_claim':False,'historical_initialization_final_eligible':False,'GPU_used':False,'new_PB_calls':0,'scaler_scope':'N1 fit-only zero variance fallback; not reusable population estimates'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--config-sha256',required=True);p.add_argument('--stage',choices=['prepare','embed','smoke'],required=True);a=p.parse_args()
    assert __debug__ and sha(a.config)==a.config_sha256;cfg=load(a.config)
    start=time.monotonic();resources=resource_preflight(cfg)
    name={'prepare':'prepared','embed':'embedding','smoke':'smoke'}[a.stage]
    out=Path(cfg['output'])/name;out.mkdir(exist_ok=False)
    put(out/'started.json',{'stage':a.stage,'code_sha256':sha(__file__),'config_sha256':a.config_sha256,'pid':os.getpid(),'time':time.time(),'resources':resources})
    try:
        result={'prepare':prepare,'embed':embed,'smoke':smoke}[a.stage](cfg,out)
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=cfg['max_RSS_bytes']
        result.update(code_sha256=sha(__file__),config_sha256=a.config_sha256,elapsed_seconds=time.monotonic()-start,max_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,python=sys.version)
        put(out/'summary.json',result);print(json.dumps(result))
    except BaseException as exc:
        put(out/'failure.json',{'error':repr(exc),'elapsed_seconds':time.monotonic()-start,'code_sha256':sha(__file__)});raise
if __name__=='__main__':main()
