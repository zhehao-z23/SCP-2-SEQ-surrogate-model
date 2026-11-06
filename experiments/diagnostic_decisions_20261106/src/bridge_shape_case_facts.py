"""Bounded facts for two frozen shape cases and prespecified matched controls.

No model loading, fitting, phi reads, homology search, or reclassification.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()


def load(p): return json.loads(Path(p).read_text())


def put(p,x):
    with Path(p).open('x') as f: json.dump(x,f,indent=2,allow_nan=False);f.write('\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);a=ap.parse_args()
    import resource
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    assert __debug__ and os.sched_getaffinity(0)=={174}
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
    plan=load(a.plan);assert sha(__file__)==plan['code_sha256'];root=Path(plan['root']);out=Path(plan['output'])
    assert not out.exists();out.mkdir(parents=True);start=time.monotonic();refs=[]
    put(out/'started.json',{'pid':os.getpid(),'unix':time.time(),'CPU':[174],'threads':1,'address_space_GiB':8,'timeout_seconds':600,'code_sha256':sha(__file__),'plan_sha256':sha(a.plan)})
    def bind(p,expected):
        p=Path(p);actual=sha(p);refs.append({'path':str(p),'sha256':actual,'expected_sha256':expected,'bytes':p.stat().st_size});assert actual==expected,str(p)
    def rowdict(row): return json.loads(row.to_frame().T.to_json(orient='records'))[0]
    def support(z):return {'rows':len(z),'unique_T':int(z.sequence_sha256.nunique()),'groups':int(z.group_id.nunique())}
    try:
        data=root/'atlas_scale_20261105/data';supp=root/'bridge_20261106/diagnostic_supplements'
        bind(data/'release.json',plan['release_sha256']);release=load(data/'release.json')
        bind(data/'samples.parquet',release['artifacts']['samples.parquet'])
        columns=['sample_id','sequence_sha256','length','group_id','fold','source','coverage','target_row','source_structure_sha256','source_origin']
        meta=pq.read_table(data/'samples.parquet',columns=columns).to_pandas();assert meta.sample_id.is_unique
        role_root=root/'bridge_20261106/roles';bind(role_root/'code/bridge_roles_guard.py',plan['guard_code_sha256'])
        sys.path.insert(0,str(role_root/'code'));from bridge_roles_guard import check
        req={'stage':'development','sequence_sha256':sorted(meta.sequence_sha256.unique().tolist()),'aliases':[]}
        put(out/'role_preflight.json',{'request':req,'receipts':[check(role_root/'registry_v1/guard_contract.json',plan['guard_contract_sha256'],dict(operation=op,**req)) for op in ['read_target','model_eval']]})
        # No packet target, QC/count, prediction, or mean-embedding payload precedes guard.
        packet_path=Path(plan['packet']['path']);bind(packet_path,plan['packet']['sha256']);packet=load(packet_path)
        selected={t:next(x['selected_sample_id'] for x in packet['type_selections'] if x['type']==t) for t in plan['case_types']}
        assert len(set(selected.values()))==2
        original_cases={x['sample_id']:x for x in packet['cases']}
        for sid in selected.values():assert sid in original_cases
        base=supp/'shape_model_comparison_v1';bind(base/'summary.json',plan['OOF_summary_sha256']);summary=load(base/'summary.json')
        bind(base/'identity_and_prediction_replay.json',summary['identity_sha256']);identity=load(base/'identity_and_prediction_replay.json')
        pin_by_path={r['path']:r['sha256'] for r in identity['refs']}
        bind(base/'per_sample_metrics.parquet',summary['private_outputs']['per_sample_metrics']['sha256'])
        per=pq.read_table(base/'per_sample_metrics.parquet',filters=[('family','=','residue_mean'),('seed','=',1701)]).to_pandas().set_index('sample_id').loc[meta.sample_id].reset_index()
        assert np.array_equal(per.sequence_sha256,meta.sequence_sha256) and np.array_equal(per.group_id,meta.group_id)
        rep=supp/'representation_audit_v1';bind(rep/'summary.json',plan['representation_summary_sha256']);rs=load(rep/'summary.json')
        bind(rep/'identity.json',rs['identity_sha256']);ri=load(rep/'identity.json')
        for r in ri['refs']:bind(r['path'],r['sha256'])
        bind(rep/'per_sample.parquet',rs['private_table']['sha256'])
        raw=pq.read_table(rep/'per_sample.parquet').to_pandas().set_index('sample_id').loc[meta.sample_id].reset_index()
        assert np.array_equal(raw.q50_raw_QC,per.q50_raw) and np.array_equal(raw.sequence_sha256,meta.sequence_sha256)
        d=meta.copy()
        for name in ['raw_IQR','raw_Bowley']:d[name]=raw[name]
        d['q50_raw']=raw.q50_raw_QC;d['residue_H']=per.H
        old_path=root/'data/qualification/release_v1/sample_manifest.parquet';bind(old_path,plan['old_manifest_sha256'])
        oldcols=['sample_id','parent_id','sequence_sha256','U_length','teacher_sequence_sha256','teacher_length','source_structure_sha256','source_origin','binding_status','parent_mapping_authoritative','static_eligible','label_join_valid','target_row','T_U_coverage','T_query_coverage','T_identity','missing_residue_count','missing_acidic_DE','missing_basic_KRH','strict_sequence_equal','target_file_sha256','target_source_sha256']
        old=pq.read_table(old_path,columns=oldcols,filters=[('sample_id','in',meta.sample_id.tolist())]).to_pandas().set_index('sample_id').loc[meta.sample_id].reset_index()
        assert np.array_equal(old.teacher_sequence_sha256,meta.sequence_sha256)
        assert np.array_equal(old.teacher_length,meta.length) and np.array_equal(old.target_row,meta.target_row)
        assert np.allclose(old.T_U_coverage,meta.coverage,equal_nan=True,atol=0,rtol=0)
        assert np.array_equal(old.source_structure_sha256,meta.source_structure_sha256)
        assert old.static_eligible.all() and old.label_join_valid.all()
        relation=np.where(old.sequence_sha256.isna(),'U unavailable',np.where(old.sequence_sha256==meta.sequence_sha256,'T = U','T differs U'))
        d['T_U_relation']=relation
        controls=[];cases=[];positions=pd.Series(np.arange(len(meta)),index=meta.sample_id)
        for typ,sid in selected.items():
            i=int(positions[sid]);row=d.iloc[i];samefold=d.fold==row.fold
            median=float(d.loc[samefold,'residue_H'].median());lr=d.length/row.length;ir=d.raw_IQR/row.raw_IQR
            valid=(samefold&(d.source==row.source)&(d.sequence_sha256!=row.sequence_sha256)&d.coverage.notna()&pd.notna(row.coverage)&(abs(d.coverage-row.coverage)<=.1)&(lr>=.8)&(lr<=1.25)&(ir>=.8)&(ir<=1.25)&(abs(d.q50_raw-row.q50_raw)<=plan['native_bin_width'])&(abs(d.raw_Bowley)<.1)&(d.residue_H<=median))
            eligible=d.loc[valid].copy();eligible['standardized_distance_squared']=(np.log(eligible.length/row.length)/np.log(1.25))**2+((eligible.coverage-row.coverage)/.1)**2+(np.log(eligible.raw_IQR/row.raw_IQR)/np.log(1.25))**2+((eligible.q50_raw-row.q50_raw)/plan['native_bin_width'])**2
            eligible['sample_id_sha256']=eligible.sample_id.map(lambda s:hashlib.sha256(s.encode()).hexdigest())
            eligible=eligible.sort_values(['standardized_distance_squared','sequence_sha256','sample_id_sha256','sample_id'])
            control=None if len(eligible)==0 else str(eligible.iloc[0].sample_id)
            controls.append({'case_type':typ,'case_sample_id':sid,'control_sample_id':control,'eligible':support(eligible),'fold_residue_H_row_median':median,'missing_case_coverage':bool(pd.isna(row.coverage)),'same_group_as_case':bool(control and d.iloc[int(positions[control])].group_id==row.group_id),'distance_squared':None if control is None else float(eligible.iloc[0].standardized_distance_squared),'no_rule_relaxation':True})
            eligible.to_parquet(out/(typ+'_eligible_controls_private.parquet'),index=False)
            cases.append((typ,sid,'frozen_badcase'))
            if control is not None:cases.append((typ,control,'matched_normal_control'))
        chosen_ids=[sid for _,sid,_ in cases];assert len(chosen_ids)==len(set(chosen_ids))
        # Fresh per-case raw count/QC/index closure, without reading any phi matrix.
        qcpath=root/'source/labels/source_snapshots/matrix_phi_qc.parquet';countpath=root/'source/labels/source_snapshots/vectors_q05_995_bins63_raw_count.parquet';idxpath=root/'data/labels/index.parquet'
        qcols=['sample_id','q25','q50','q75','n_rows','n_vertices','n_nan_phi','n_inf_phi','phi_column','matrix_columns','matrix_path','readable']
        qc=pq.read_table(qcpath,columns=qcols,filters=[('sample_id','in',chosen_ids)]).to_pandas().set_index('sample_id').loc[chosen_ids]
        ct=pq.read_table(countpath,filters=[('sample_id','in',chosen_ids)]).to_pandas().set_index('sample_id').loc[chosen_ids]
        li=pq.read_table(idxpath,filters=[('sample_id','in',chosen_ids)]).to_pandas().set_index('sample_id').loc[chosen_ids]
        counts=ct[[f'bin_{j:03d}' for j in range(63)]].to_numpy();assert np.issubdtype(counts.dtype,np.integer) and (counts>=0).all()
        sums=counts.sum(1);assert np.array_equal(sums,qc.n_rows) and np.array_equal(sums,qc.n_vertices) and np.array_equal(sums,li.n_vertices)
        assert qc.readable.all() and qc.phi_column.eq('phi').all() and qc.matrix_columns.eq('phi,x,y,z').all() and qc.n_nan_phi.eq(0).all() and qc.n_inf_phi.eq(0).all() and li.label_integrity_valid.all()
        assert np.array_equal(li.row,meta.iloc[positions.loc[chosen_ids]].target_row)
        y=np.load(data/'p.npy',allow_pickle=False);assert np.array_equal((counts/sums[:,None]).astype(np.float32),y[positions.loc[chosen_ids]])
        assert np.array_equal(qc[['q25','q50','q75']].to_numpy(),raw.iloc[positions.loc[chosen_ids]][['q25_raw_QC','q50_raw_QC','q75_raw_QC']].to_numpy())
        edges=np.asarray(load(root/'data/labels/release.pending.json')['edges']);assert np.allclose(np.diff(edges),plan['native_bin_width'],atol=1e-12,rtol=0)
        fitfile=root/'overnight_20261106/confirmation/results/all_fits.csv';bind(fitfile,plan['fit_index_sha256']);fits=pd.read_csv(fitfile)
        predicted={};roles_by_fold={};fit_refs=[]
        for fold in sorted({int(meta.iloc[positions[sid]].fold) for sid in chosen_ids}):
            for family in ['composition_charge_deep','mean_pre','residue_mean']:
                ff=fits[(fits.fold==fold)&(fits.seed==1701)&(fits.family==family)];assert len(ff)==1
                run=Path(ff.iloc[0].artifact_path)
                for name in ['receipt.json','roles.json','evaluation_prediction.npy']:bind(run/name,pin_by_path[str(run/name)])
                rec=load(run/'receipt.json');roles=load(run/'roles.json');p=np.load(run/'evaluation_prediction.npy',allow_pickle=False)
                assert rec['prediction_sha256']==pin_by_path[str(run/'evaluation_prediction.npy')]
                ei=positions.loc[roles['evaluation_ids']].to_numpy();assert np.array_equal(ei,np.flatnonzero(meta.fold.to_numpy()==fold))
                assert p.shape==(len(ei),63) and np.isfinite(p).all() and (p>=0).all() and abs(p.sum(1)-1).max()<1e-6
                train=meta.set_index('sample_id').loc[roles['train_ids']];assert set(train.fold)==set(range(5))-{fold,(fold+1)%5}
                assert not set(train.group_id)&set(meta.iloc[ei].group_id)
                if fold in roles_by_fold:assert roles_by_fold[fold]==roles
                roles_by_fold[fold]=roles
                for sid in chosen_ids:
                    if int(meta.iloc[positions[sid]].fold)==fold:
                        v=p[roles['evaluation_ids'].index(sid)];predicted[(sid,family)]=v
                        if sid in selected.values():assert np.array_equal(v,np.asarray(original_cases[sid]['curves'][family]['p'],np.float32))
                fit_refs.append({'family':family,'fold':fold,'seed':1701,'run':str(run),'receipt_sha256':sha(run/'receipt.json'),'roles_sha256':sha(run/'roles.json'),'prediction_sha256':sha(run/'evaluation_prediction.npy')})
        # Frozen mean vectors: cosine is an input-space proximity diagnostic, never homology.
        cache=root/'pooling_20261105/cache'
        for name,key in [('receipt.json','cache_receipt_sha256'),('index.parquet','cache_index_sha256'),('mean.npy','cache_mean_sha256')]:bind(cache/name,plan[key])
        ci=pq.read_table(cache/'index.parquet').to_pandas();assert ci.sequence_sha256.is_unique
        mean=np.load(cache/'mean.npy',allow_pickle=False);assert mean.shape==(len(ci),960) and np.isfinite(mean).all()
        em=mean.astype(np.float64);norm=np.linalg.norm(em,axis=1);assert (norm>0).all();em/=norm[:,None]
        epos=pd.Series(np.arange(len(ci)),index=ci.sequence_sha256);assert set(meta.sequence_sha256)<=set(epos.index)
        assert np.array_equal(ci.iloc[epos.loc[meta.sequence_sha256]].length.to_numpy(),meta.length.to_numpy())
        coarse_file=Path(plan['coarse_code']['path']);bind(coarse_file,plan['coarse_code']['sha256']);sys.path.insert(0,str(coarse_file.parent));from bridge_coarse_shape_support import peaks
        records=[];private_packets=[]
        def distribution_facts(p,truth):
            p=np.asarray(p,float);truth=np.asarray(truth,float)
            density=p[1:-1]/np.diff(edges);ii=int(np.argmax(density));left=right=ii;eps=float(density.max())*1e-12
            while left>0 and abs(density[left-1]-density[ii])<=eps:left-=1
            while right+1<len(density) and abs(density[right+1]-density[ii])<=eps:right+=1
            local=peaks(density,0.)
            for v in local:
                v['interval']=[float(edges[v['left_bin']]),float(edges[v['right_bin']+1])];v['prominence_fraction_of_global_max']=v['prominence']/float(density.max());v['passes_existing_5percent_prominence']=bool(v['prominence']+1e-12*density.max()>=.05*density.max())
            inside=(edges[:-1]>=20)&(edges[1:]<=40);intersect=(edges[1:]>20)&(edges[:-1]<40)
            lengths=np.maximum(0,np.minimum(edges[1:],40)-np.maximum(edges[:-1],20))
            delta=np.asarray(p,float).cumsum()[:-1]-np.asarray(truth,float).cumsum()[:-1]
            return {'maximum_native_density_interval':[float(edges[left]),float(edges[right+1])],'maximum_interval_rule':'First argmax plateau, extended at relative1e-12 tolerance; separated equal maxima use the leftmost','maximum_density':float(density[ii]),'unthresholded_finite_interior_local_peaks':local,
                'window20_40_mass':{'lower':float(p[1:-1][inside].sum()),'upper':float(p[1:-1][intersect].sum()),'uniform_proxy':float(np.sum(p[1:-1]*lengths/np.diff(edges))),'post_visual_description':True,'no_bin_center_assignment':True},
                'negative_open_mass':float(p[0]),'positive_open_mass':float(p[-1]),'negative_open_bias_vs_true63':float(p[0]-truth[0]),
                'cutpoint_CDF_delta':{'definition':'prediction minus true63 cumulative mass at62 finite cutpoints','minimum':float(delta.min()),'maximum':float(delta.max()),'positive_over_1e8':int((delta>1e-8).sum()),'negative_below_minus1e8':int((delta< -1e-8).sum()),'near_zero_1e8':int((abs(delta)<=1e-8).sum())}}
        for typ,sid,role in cases:
            i=int(positions[sid]);row=d.iloc[i];oldrow=old.iloc[i];fold=int(row.fold);trainids=roles_by_fold[fold]['train_ids'];train=d.iloc[positions.loc[trainids]].copy()
            side='negative' if typ.endswith('negative') else 'positive'
            strong=(train.raw_IQR>=1)&((train.raw_Bowley<=-.3) if side=='negative' else (train.raw_Bowley>=.3))
            same_side=support(train.loc[strong]);all_sides={z:support(train.loc[(train.raw_IQR>=1)&((train.raw_Bowley<=-.3) if z=='negative' else (train.raw_Bowley>=.3))]) for z in ['negative','positive']}
            t=train.sort_values(['sequence_sha256','sample_id']).drop_duplicates('sequence_sha256').copy();assert row.sequence_sha256 not in set(t.sequence_sha256)
            t['cosine']=em[epos.loc[t.sequence_sha256].to_numpy()]@em[int(epos[row.sequence_sha256])]
            t=t.sort_values(['cosine','sequence_sha256'],ascending=[False,True]);top=t.head(5);neighbors=[]
            for _,n in top.iterrows():
                views=train[train.sequence_sha256==n.sequence_sha256];same=(views.raw_IQR>=1)&((views.raw_Bowley<=-.3) if side=='negative' else (views.raw_Bowley>=.3))
                neighbors.append({'sequence_sha256':str(n.sequence_sha256),'canonical_sample_id':str(n.sample_id),'group_id':str(n.group_id),'cosine':float(n.cosine),'views':len(views),'same_side_strong_any_view':bool(same.any()),'raw_Bowley_min':float(views.raw_Bowley.min()),'raw_Bowley_max':float(views.raw_Bowley.max())})
            true=y[i];assert np.array_equal(true,np.asarray(original_cases[sid]['target_p'],np.float32)) if sid in selected.values() else True
            facts={'case_type':typ,'role':role,'metadata':rowdict(row),'T_U_details':rowdict(oldrow),'raw_QC':rowdict(qc.loc[sid]),'identity':{'target_row_matches_label_index':True,'counts_sum_equals_QC_vertices_and_rows':True,'count_normalization_matches_current_FP32_p':True,'raw_quartiles_match_representation_audit':True,'teacher_hash_length_source_match_current_meta':True},
                'train_support':{'folds':sorted(map(int,train.fold.unique())),'cohort':support(train),'same_side':same_side,'both_sides':all_sides},
                'embedding_neighbors':{'definition':'Cosine of raw frozen cached mean vectors, L2 normalized; not homology or a distance in train-standardized MLP features','k':5,'training_T_count':len(t),'neighbors':neighbors,'same_side_strong_T_in_top5':sum(n['same_side_strong_any_view'] for n in neighbors),'no_evaluation_or_stop_neighbor':True},
                'truth63':distribution_facts(true,true),'models':{},'frozen_coarse_details':original_cases[sid]['coarse_details'] if sid in selected.values() else None}
            for family in ['composition_charge_deep','mean_pre','residue_mean']:
                pred=predicted[(sid,family)];facts['models'][family]=distribution_facts(pred,true)
                facts['models'][family]['H']=float(np.sqrt(.5*((np.sqrt(true.astype(float))-np.sqrt(pred.astype(float)))**2).sum()))
            if sid in selected.values():assert abs(facts['models']['residue_mean']['H']-float(row.residue_H))<=1e-7
            private_packets.append({'sample_id':sid,'true_p':true.tolist(),'predicted_p':{f:predicted[(sid,f)].tolist() for f in ['composition_charge_deep','mean_pre','residue_mean']},'counts':counts[chosen_ids.index(sid)].tolist(),**facts})
            # Small private facts omit raw matrix paths and individual neighbor identifiers.
            light=json.loads(json.dumps(facts));light['raw_QC'].pop('matrix_path',None)
            light['embedding_neighbors']['neighbors']=[{k:v for k,v in n.items() if k not in ['sequence_sha256','canonical_sample_id','group_id']} for n in neighbors]
            records.append(light)
        put(out/'private_case_packet.json',{'cases':private_packets,'matching':controls,'plan_sha256':sha(a.plan)})
        put(out/'identity.json',{'status':'passed','refs':refs,'fit_references':fit_refs,'no_phi_or_sequence_text_read':True})
        result={'schema':'shape_case_facts_v1','status':'complete','cases':records,'matching':controls,'normal_matching_frozen_before_execution':True,'no_eligible_case_replacement':True,
            'limits':['Two frozen descriptive examples, not prevalence or independent causal evidence.','Matched control may share homology group; this is explicitly marked and is not independent-group evidence.','Top5 mean-embedding proximity is not homology, structural similarity, or causal support.','Unthresholded local peaks describe original finite-bin density using existing plateau/boundary handling; original native/three-view classifications unchanged.','Window20_40 was chosen after viewing case06; report fraction bounds and labeled uniform proxy, never bin-center exactness.','Existing one_major_peak means one supported major basin under that coarse rule, not absence of all small peaks or raw multimodality.'],
            'code_sha256':sha(__file__),'plan_sha256':sha(a.plan),'identity_sha256':sha(out/'identity.json'),'private_packet_sha256':sha(out/'private_case_packet.json'),'resource':{'CPU':[174],'threads':1,'address_space_GiB':8,'elapsed_seconds':time.monotonic()-start,'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},'new_fits':0,'model_loads':0,'raw_phi_reads':0}
        put(out/'private_facts.json',result);print(json.dumps({'status':'complete','cases':len(records),'normal_controls':sum(x['control_sample_id'] is not None for x in controls),'facts_sha256':sha(out/'private_facts.json'),'elapsed_seconds':result['resource']['elapsed_seconds']}))
    except Exception as exc:
        put(out/'failure.json',{'status':'failed_preserved','error':str(exc),'traceback':traceback.format_exc(),'refs_so_far':refs});raise


if __name__=='__main__':main()
