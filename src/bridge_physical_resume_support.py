"""Read-only reconstruction and complete reuse verification; never solves or samples."""
from pathlib import Path
import json,sqlite3
import numpy as np
import bridge_physical_run as b

def pending_rows(metadata,cached):
    expected={x['rank']for x in metadata};done={x['rank']for x in cached}
    assert len(expected)==len(metadata)and len(done)==len(cached)and done<=expected
    return [x for x in metadata if x['rank']not in done]

def merge_results(metadata,cached,fresh):
    results=cached+fresh
    assert len(results)==len(metadata)and len({x['rank']for x in results})==len(results)and {x['rank']for x in results}=={x['rank']for x in metadata}
    return results

def frozen_plan(sid):
    d=b.R/'cases'/sid/'further';summary=b.load(d/'nested_plan.json');pre=b.load(b.R/'cases'/sid/'prepared_case.json')
    pqr=Path(pre['pqr_path']);mesh=Path(pre['mesh_root']);q=np.load(mesh/'vertices.npy');nn=np.load(b.R/'cases'/sid/'overlap12/geometry/surface_normals.npy');w=np.load(mesh/'vertex_area_weights.npy')
    assert summary['same_PQR']['sha256']==b.sha(pqr)and summary['mesh_manifest']['sha256']==b.sha(mesh/'surface_manifest.json')and summary['physical_object']==pre['source']
    assert summary['parent_h_A']==.125 and summary['fine_h_A']==.0625 and summary['fine_halo_A']==8.
    # Reconstruct only indices/geometry using the unchanged frozen planner.
    # Compare before any file write; do not overwrite old plans or indices.
    atoms=b.read_pqr_atoms(pqr);fine=b.fine_box(atoms[['x','y','z']].values,atoms.radius.values,q,h=.0625)
    parent_plan=b.load(b.R/'cases'/sid/'overlap12/geometry_plan.json');parent_fine=parent_plan['fine']
    parents=b.query_box_plan(q,parent_fine['origin'],.125,halo_A=12.,cap_bytes=80*1024**3,max_points=100000000)
    rows=[];owned=np.zeros(len(q),np.int32)
    for parent in parents:
        pidx=parent['query_indices'];children=b.query_box_plan(q[pidx],fine['origin'],.0625,halo_A=8.,cap_bytes=80*1024**3,max_points=100000000)
        for child in children:
            row=summary['rows'][len(rows)];idx=np.load(d/('query_indices_'+str(row['rank'])+'.npy'))
            assert np.array_equal(idx,pidx[child['query_indices']])
            assert row['rank']==len(rows)and row['parent_rank']==parent['rank']and row['parent_source_key']==sid+'_overlap12_fine_'+str(parent['rank'])
            assert row['parent_query_plan']=={k:v for k,v in parent.items()if k!='query_indices'}
            origin=np.array(child['global_origin'])+np.array(child['compute_start'])*.0625
            assert row['local_dime']==child['local_dime']and np.array_equal(origin,np.array(row['origin']))and np.array_equal((np.array(child['local_dime'])-1)*.0625,np.array(row['length']))
            np.add.at(owned,idx,1);rows.append({**row,'query_indices':idx})
    assert len(rows)==len(summary['rows'])==68 and np.all(owned==1)and len(q)==summary['full_query_count']
    return summary,rows,pre,pqr,mesh,q,nn,w

class ReuseVerifier:
    def __init__(self,sid,heavy):
        self.sid=sid;self.heavy=Path(heavy);self.hashes={};self.native={};self.pre=b.load(b.R/'cases'/sid/'prepared_case.json');self.pqr=Path(self.pre['pqr_path']);self.mesh=Path(self.pre['mesh_root']);self.physical=b.load(b.P/'config/production_candidate_SES_v2.json')['PB_physical_parameters'];self.pqr_sha=self.digest(self.pqr)
        self.vertices=np.load(self.mesh/'vertices.npy');self.runtime=b.solver.runtime();self.runtime_binding=self.runtime.binding();self.solver_sha=self.digest(b.P/'code/SES_map_solver_v3_runtime.py');self.apbs_sha=self.digest(self.runtime.tool_path('apbs'))
    def digest(self,path):
        path=str(path)
        if path not in self.hashes:self.hashes[path]=b.sha(path)
        return self.hashes[path]
    def ref(self,item):
        assert self.digest(item['path'])==item['sha256'],item['path'];return Path(item['path'])
    def field(self,path):
        path=Path(path)
        if str(path)in self.native:return self.native[str(path)]
        field=b.load(path);src=field['source'];assert field['status']=='completed'and field['logs']['logs_acceptable']
        with sqlite3.connect('file:'+str(b.R/'records/budget/PB_attempts.sqlite')+'?mode=ro',uri=True)as db:row=db.execute('select status,result_path,input_hash from attempts where id=?',(field['PB_attempt_id'],)).fetchone()
        assert row and row[0]=='completed'and Path(row[1])==path
        import hashlib
        assert row[2]==hashlib.sha256(json.dumps(src,sort_keys=True).encode()).hexdigest()
        assert src['source_PQR']==str(self.pqr)and src['source_PQR_sha256']==self.pqr_sha and src['physical']==self.physical
        assert src['native_runtime']==self.runtime_binding and src['APBS_sha256']==self.apbs_sha and src['code_sha256']==self.solver_sha
        cwd=Path(field['result']['cwd']);assert self.digest(cwd/'molecule.pqr')==src['normalized_PQR_sha256']and self.digest(cwd/'apbs.in')==src['input_sha256']
        assert self.digest(field['dx_path'])==field['dx_sha256']and field['result']['returncode']==0 and not field['result']['timed_out']and field['result']['threads']==1
        if src['boundary_map']:assert self.digest(src['boundary_map'])==src['boundary_map_sha256']
        maps=src['coefficient_maps'];assert len(maps)==4
        for item in maps:self.ref(item)
        expected=b.solver.manual_input('molecule.pqr',src['dims'],src['length'],src['center'],self.physical,src['boundary_map'],[x['path']for x in maps])
        assert (cwd/'apbs.in').read_text()==expected
        manifest=b.load(Path(maps[0]['path']).parent/'map_manifest.json');ms=manifest['source']
        assert manifest['maps']==maps and ms['PQR_sha256']==self.pqr_sha and ms['vertices_sha256']==self.digest(self.mesh/'vertices.npy')and ms['faces_sha256']==self.digest(self.mesh/'faces.npy')
        assert ms['dims']==src['dims']and np.allclose(ms['origin_A'],field['origin'],rtol=0,atol=5e-5)and np.allclose(ms['h_A'],field['h'],rtol=0,atol=5e-6)
        archive=field['archive'];assert archive['lossless_roundtrip']and archive['decompressed_sha256']==field['dx_sha256'];self.ref(archive['compressed'])
        self.native[str(path)]=field;return field
    def sampling(self,row):
        rank=row['rank'];key=self.sid+'_further_nested_fine_'+str(rank);part=self.heavy/'parts'/key;path=part/'sampling_receipt.json';sample=b.load(path)
        expected={'field':self.heavy/'records/SES_map_fields'/(key+'.json'),'parent_field':b.R/'records/SES_map_fields'/(row['parent_source_key']+'.json'),'query_indices':b.R/'cases'/self.sid/'further'/('query_indices_'+str(rank)+'.npy'),'phi':part/'phi_jump.npy','raw':part/'phi_raw.npy','accepted':part/'query_accepted.npy'}
        assert sample['rank']==rank and sample['key']==key and sample['APBS_threads']==1
        for name,p in expected.items():assert self.ref(sample[name])==p
        field=self.field(expected['field']);parent=self.field(expected['parent_field']);src=field['source'];center=np.array(row['origin'])+np.array(row['length'])/2
        assert src['dims']==row['local_dime']and src['length']==row['length']and src['center']==center.tolist()and src['boundary_map']==parent['dx_path']and src['boundary_map_sha256']==parent['dx_sha256']
        assert [x['path']for x in src['coefficient_maps']]==[str(self.heavy/'maps'/key/name)for name in ['diel0.dx','diel1.dx','diel2.dx','kappa.dx']]
        pp=row['parent_query_plan'];po=np.array(pp['global_origin'])+np.array(pp['compute_start'])*.125
        assert parent['dims']==pp['local_dime']and np.allclose(parent['h'],.125,rtol=0,atol=5e-6)and np.allclose(parent['origin'],po,rtol=0,atol=5e-5)
        margin=np.minimum(np.array(row['origin'])-np.array(parent['origin']),np.array(parent['origin'])+(np.array(parent['dims'])-1)*np.array(parent['h'])-np.array(row['origin'])-np.array(row['length']))
        assert np.all(margin>=.125-1e-5)and np.allclose(margin,sample['actual_parent_margin_A'],rtol=0,atol=1e-12)
        idx=np.load(expected['query_indices']);phi=np.load(expected['phi']);raw=np.load(expected['raw']);accepted=np.load(expected['accepted'])
        assert np.issubdtype(idx.dtype,np.integer)and idx.ndim==1 and len(idx)>0 and len(np.unique(idx))==len(idx)and idx.min()>=0 and idx.max()<len(self.vertices)
        assert phi.shape==raw.shape==accepted.shape==idx.shape and accepted.dtype==np.bool_ and np.isfinite(phi).all()and np.isfinite(raw).all()and np.array_equal(phi[~accepted],raw[~accepted])
        lo=np.floor((self.vertices[idx]-np.array(field['origin']))/np.array(field['h'])).astype(np.int64);assert np.all(lo>=0)and np.all(lo+1<np.array(field['dims']))
        return sample,{'rank':rank,'sampling_receipt':b.ref(path),'native_field':sample['field'],'parent_field':sample['parent_field'],'full_native_source_and_maps_verified':True,'complete_sampling_arrays_verified':True,'not_resampled':True}
