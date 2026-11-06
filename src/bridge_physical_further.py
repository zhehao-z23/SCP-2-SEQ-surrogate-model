"""Full AF h0.0625 validation focused through verified h0.125 overlap12 fields.

The complete area distribution is retained. Child boxes must be strictly within
their own intermediate parent DX; no clipped query, partial mesh, sampled proxy,
or direct coarse0.5-to-fine0.0625 boundary is accepted.
"""
from pathlib import Path
import argparse,json,sys,os,time,shutil
sys.dont_write_bytecode=True
import bridge_physical_run as b
import numpy as np

def plan(sid):
    pre=b.prepare(sid);od,pqr,mesh,q,nn,w,ph,parent,parents=b.geometry(pre,'overlap12')
    atoms=b.read_pqr_atoms(pqr);h=.0625;fine=b.fine_box(atoms[['x','y','z']].values,atoms.radius.values,q,h=h)
    rows=[];ownership=np.zeros(len(q),np.int8)
    for pp in parents:
        pidx=pp['query_indices'];po=np.array(pp['global_origin'])+np.array(pp['compute_start'])*ph;pu=po+(np.array(pp['local_dime'])-1)*ph
        children=b.query_box_plan(q[pidx],fine['origin'],h,halo_A=8.,cap_bytes=80*1024**3,max_points=100000000)
        for child in children:
            idx=pidx[child['query_indices']];co=np.array(child['global_origin'])+np.array(child['compute_start'])*h;cu=co+(np.array(child['local_dime'])-1)*h
            margin=np.minimum(co-po,pu-cu);inside=bool(np.all(margin>=ph-1e-8))
            if not inside:raise RuntimeError('further child lacks one intermediate-parent grid step of boundary support: '+str((pp['rank'],child['rank'],margin.tolist())))
            ownership[idx]+=1
            rows.append({'rank':len(rows),'parent_rank':pp['rank'],'parent_query_plan':{k:v for k,v in pp.items()if k!='query_indices'},'query_indices':idx,'local_dime':child['local_dime'],'origin':co.tolist(),'length':((np.array(child['local_dime'])-1)*h).tolist(),'parent_minimum_margin_A':margin.tolist(),'minimum_stencil_margin_A':child['minimum_stencil_margin_A'],'reservation_scenario_bytes':child['reservation_scenario_bytes'],'parent_source_key':sid+'_overlap12_fine_'+str(pp['rank'])})
    assert np.all(ownership==1)
    d=b.R/'cases'/sid/'further';d.mkdir(exist_ok=True);metadata=[]
    for row in rows:
        np.save(d/('query_indices_'+str(row['rank'])+'.npy'),row['query_indices']);metadata.append({k:v for k,v in row.items()if k!='query_indices'})
    costs=[]
    for rr in (b.R/'records/SES_map_fields').glob(sid+'_refined_fine_*.json'):
        r=b.load(rr)
        if r.get('status')!='completed':continue
        nodes=int(np.prod(r['dims']));maps=r['source']['coefficient_maps'];bytes_total=sum(Path(x['path']).stat().st_size for x in maps)+Path(r['dx_path']).stat().st_size+Path(r['dx_path']+'.zst').stat().st_size
        costs.append({'receipt':b.ref(rr),'nodes':nodes,'retained_field_plus_map_bytes':bytes_total,'bytes_per_node':bytes_total/nodes,'APBS_seconds':r['result']['wall_seconds'],'APBS_seconds_per_node':r['result']['wall_seconds']/nodes})
    points=sum(int(np.prod(row['local_dime']))for row in rows);disk=shutil.disk_usage(b.R)
    with b.sqlite3.connect('file:'+str(b.R/'records/budget/PB_attempts.sqlite')+'?mode=ro',uri=True)as db:used=db.execute('select count(*)from attempts').fetchone()[0]
    summary={'schema':'AF_full_further_nested_reference_plan_v1','case':sid,'parent_variant':'overlap12','parent_h_A':ph,'fine_h_A':h,'fine_halo_A':8.,'full_query_count':len(q),'queries_owned_exactly_once':True,'minimum_parent_containment_margin_A':float(min(min(r['parent_minimum_margin_A'])for r in rows)),'parent_interpolation_support_grid_steps_min':float(min(min(r['parent_minimum_margin_A'])for r in rows)/ph),'new_PB_attempts':len(rows),'existing_ledger_attempts_at_plan':used,'physical_object':pre['source'],'same_PQR':b.ref(pqr),'mesh_manifest':b.ref(mesh/'surface_manifest.json'),'further_fine_box':fine,'rows':metadata,'cost_evidence':costs,'total_grid_nodes':points,'disk_free_bytes':disk.free,'predicted_incremental_retained_bytes':None if not costs else int(points*max(c['bytes_per_node']for c in costs)*1.25),'predicted_APBS_seconds':None if not costs else points*max(c['APBS_seconds_per_node']for c in costs)*1.5,'new_PB_launched_by_plan':0,'qualified':False}
    b.write(d/'nested_plan.json',summary);return summary,rows,pre,pqr,mesh,q,nn,w

# Only this planning function was used by the frozen parallel executor.
# The inactive historical serial execution path is intentionally absent.
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',required=True)
    args=parser.parse_args();summary,*_=plan(args.case)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()