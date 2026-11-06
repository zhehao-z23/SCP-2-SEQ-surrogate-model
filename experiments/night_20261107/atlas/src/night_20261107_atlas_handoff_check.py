"""Read-only checks of completed N1 adapter artifacts; no fit or model prediction."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from night_20261107_atlas_entry import load,sha,put,verified,authority,resource_preflight,make_model,transfer_hidden

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args();cfg=load(a.config);resource_preflight(cfg)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);root=Path(cfg['output'])
    bind=load(root/'prepared/prepared_private.json');authority(cfg,'predict',bind['sequence_sha256'])
    for name,pin in bind['files'].items():assert sha(root/'prepared'/name)==pin
    cr=load(root/'embedding/cache_receipt.private.json');assert sha(root/'embedding/residues.private.npy')==cr['flat_sha256']
    x=np.load(root/'embedding/residues.private.npy').astype(np.float32)[None];ch=np.load(root/'prepared/charge.private.npy')
    old=torch.load(verified(cfg['historical_checkpoint']),map_location='cpu',weights_only=False)
    astate=torch.load(root/'smoke/from_scratch.private.pt',map_location='cpu',weights_only=False)
    bstate=torch.load(root/'smoke/historical_hidden_rebased.private.pt',map_location='cpu',weights_only=False)
    new=astate['scalers'];assert all(np.array_equal(v,bstate['scalers'][k]) for k,v in new.items())
    original=make_model('residue_mean',63).eval();original.load_state_dict(old['state_dict'])
    transformed=make_model('residue_mean',63).eval();transfer_hidden(transformed,old,new,'residue_mean')
    def first(model,scales):
        e=np.concatenate([(np.where(np.isnan(ch),scales['charge_median'],ch)-scales['charge_mean'])/scales['charge_std'],np.isnan(ch).astype(np.float32)],1)
        xx=torch.tensor((x-scales['embedding_mean'])/scales['embedding_std'])
        return torch.nn.functional.linear(xx,model.first.weight[:,:960])+torch.nn.functional.linear(torch.tensor(e),model.first.weight[:,960:],model.first.bias)[:,None]
    with torch.no_grad():error=float((first(original,old)-first(transformed,new)).abs().max())
    assert error<1e-5
    for k in ['head.1.weight','head.1.bias','head.4.weight','head.4.bias']:assert torch.equal(transformed.state_dict()[k],old['state_dict'][k])
    stats={key:sum(v.numel() for k,v in original.state_dict().items() if k.startswith(key)) for key in ['first.','head.1.','head.4.','head.7.']}
    exposure=load(root/'smoke/historical_exposure.private.json')['partitions']
    exposure_counts={k:{'unique_sequences':len({x['sequence_sha256'] for x in v}),'groups':len({x['group_id'] for x in v})} for k,v in exposure.items()}
    summaries={name:load(root/name/'summary.json') for name in ['prepared','embedding','smoke']}
    assert all(s['status']=='complete' for s in summaries.values())
    r={'status':'passed','checks_source_sha256':sha(__file__),'executed_adapter_sha256':summaries['smoke']['code_sha256'],'actual_input_first_affine_rebase_max_abs_difference':error,'layer_parameter_counts':stats,'hidden_source_derived_parameters':stats['first.']+stats['head.1.']+stats['head.4.'],'reset_output_parameters':stats['head.7.'],'historical_exposure_counts':exposure_counts,'historical_source_eligibility':'development_only; new-release final closure unresolved','no_fitting_or_model_predictions_in_this_check':True,'initialization_winner_selected':False,'source_summary_sha256':{name:sha(root/name/'summary.json') for name in summaries},'PB_calls':0,'GPU_used':False}
    put(a.output,r);print(json.dumps(r))
if __name__=='__main__':main()
