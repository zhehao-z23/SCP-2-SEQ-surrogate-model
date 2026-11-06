"""Synthetic adapter checks; no biological sequences, targets or checkpoints."""
import argparse,copy,json,time
from pathlib import Path
import numpy as np
import torch
from night_20261107_atlas_entry import make_model,transfer_hidden,validate_label_contract,final_release_gate,resource_preflight,load,sha,put

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    cfg=load(a.config);resource_preflight(cfg);torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(11);rng=np.random.default_rng(31)
    def scalers():return {'embedding_mean':rng.normal(size=960).astype('float32'),'embedding_std':rng.uniform(.3,2,960).astype('float32'),'charge_mean':rng.normal(size=39).astype('float32'),'charge_std':rng.uniform(.3,2,39).astype('float32'),'charge_median':rng.normal(size=39).astype('float32')}
    old,new=scalers(),scalers();raw=rng.normal(size=(4,7,960)).astype('float32');ch=rng.normal(size=(4,39)).astype('float32');missing=rng.random(ch.shape)<.35
    def extra(s):return torch.tensor(np.concatenate([(np.where(missing,s['charge_median'],ch)-s['charge_mean'])/s['charge_std'],missing.astype('float32')],1))
    rows=[]
    for mode in ['residue_mean','mean_pre','charge_deep']:
        model=make_model(mode,63).eval();fresh=make_model(mode,63).eval();bundle=dict(old,state_dict=model.state_dict());transfer_hidden(fresh,bundle,new,mode)
        aextra,bextra=extra(old),extra(new);xa=torch.tensor((raw-old['embedding_mean'])/old['embedding_std']);xb=torch.tensor((raw-new['embedding_mean'])/new['embedding_std'])
        with torch.no_grad():
            if mode=='charge_deep':a0=model.first(aextra);b0=fresh.first(bextra)
            elif mode=='mean_pre':a0=model.first(torch.cat([xa.mean(1),aextra],1));b0=fresh.first(torch.cat([xb.mean(1),bextra],1))
            else:
                a0=torch.nn.functional.linear(xa,model.first.weight[:,:960])+torch.nn.functional.linear(aextra,model.first.weight[:,960:],model.first.bias)[:,None]
                b0=torch.nn.functional.linear(xb,fresh.first.weight[:,:960])+torch.nn.functional.linear(bextra,fresh.first.weight[:,960:],fresh.first.bias)[:,None]
            error=float((a0-b0).abs().max());assert error<5e-6
        rows.append({'mode':mode,'mixed_missing_affine_rebase_max_abs_difference':error,'new_output_layer_retained':True})
    m=make_model('residue_mean',17).eval();x=torch.randn(2,5,960);e=torch.randn(2,78);mask=torch.ones(2,5,dtype=torch.bool);mask[0,3:]=False
    with torch.no_grad():
        q=m(x,e,mask);xx=torch.cat([x,torch.randn(2,3,960)*100],1);mm=torch.cat([mask,torch.zeros(2,3,dtype=torch.bool)],1);err=float((q-m(xx,e,mm)).abs().max());assert err<1e-6 and q.shape==(2,17)
    c={'measure':'area','equation':'npbe','potential_unit':'kBT/e','primary_array':'corrected_area_p','bin_source_sha256':'newbin','protocol_sha256':'newprotocol','sequence_view':'prepared_observed','input_matches_prepared':True,'role':'development_only','qualification_status':'accepted','n_bins':3,'internal_edges':[-1.,1.]}
    assert validate_label_contract(c,'newbin','newprotocol')
    rejected=[]
    for key,value in [('measure','vertex_count'),('equation','lpbe'),('bin_source_sha256','old63'),('input_matches_prepared',False),('role','protected'),('qualification_status','legacy_false')]:
        bad=dict(c);bad[key]=value
        try:validate_label_contract(bad,'newbin','newprotocol')
        except AssertionError:rejected.append(key)
        else:raise RuntimeError('bad label admitted')
    try:final_release_gate({'stage':'final_release','all_closure_ready':True})
    except PermissionError:rejected.append('final_release')
    else:raise RuntimeError('final release was admitted')
    result={'status':'passed','synthetic_only':True,'source_sha256':sha(__file__),'adapter_sha256':sha(Path(__file__).with_name('night_20261107_atlas_entry.py')),'affine_rebase':rows,'padding_max_abs_difference':err,'dynamic_output_bins_tested':17,'rejected_contract_mutations':rejected,'old_missing_median_changes_tested':True,'N1_scientific_result':False}
    put(a.output,result);print(json.dumps(result))
if __name__=='__main__':main()
