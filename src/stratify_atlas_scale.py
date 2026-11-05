"""Retrospective strict-T=U strata and existing-search scope audit.

This does not run a new homology search or impute undetected identity as zero.
"""
import argparse, hashlib, json
from pathlib import Path
import pandas as pd
import numpy as np

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()

def headers(path):
    with Path(path).open() as f:return {line[1:].split()[0] for line in f if line.startswith('>')}

def metrics(frame):
    cols=['H','cutpoint_CDF_MSE','edge_bin_mass_MAE']
    if len(frame)==0:return dict(samples=0,proteins=0,groups=0)
    protein=frame.groupby(['group_id','protein_id'])[cols].mean()
    group=protein.groupby('group_id').mean()
    return dict(samples=len(frame),proteins=len(protein),groups=len(group),
        **{'group_macro_'+c:float(group[c].mean()) for c in cols},
        **{'protein_macro_'+c:float(protein[c].mean()) for c in cols})

def paired_comparisons(frame, by):
    rows=[]
    for label,sub in frame.groupby(by,observed=True):
        scores={}
        for fam in ['composition_charge','esmc_charge','composition_charge_deep']:
            current=sub[sub.family==fam]
            if len(current):
                scores[fam]=current.groupby(['group_id','protein_id']).H.mean().groupby('group_id').mean()
        for left,right in [('esmc_charge','composition_charge'),('composition_charge_deep','composition_charge'),('esmc_charge','composition_charge_deep')]:
            if left not in scores or right not in scores:continue
            joined=pd.concat({left:scores[left],right:scores[right]},axis=1).dropna()
            assert len(joined)==len(scores[left])==len(scores[right])
            diff=(joined[left]-joined[right]).to_numpy()
            rng=np.random.default_rng(3407)
            bs=np.asarray([rng.choice(diff,size=len(diff),replace=True).mean() for _ in range(2000)])
            rows.append(dict(stratum=str(label),groups=len(diff),contrast=left+' minus '+right,
                group_macro_H_difference=float(diff.mean()),descriptive_group_bootstrap_low=float(np.quantile(bs,.025)),
                descriptive_group_bootstrap_high=float(np.quantile(bs,.975)),resamples=2000,seed=3407,
                interpretation='Post-selection retrospective exploratory interval; no selection or multiplicity correction; not equivalence or blind-confirmation evidence'))
    return pd.DataFrame(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--project',required=True);a=p.parse_args()
    root=Path(a.root);project=Path(a.project);out=root/'results/strata_audit';out.mkdir(parents=True,exist_ok=True)
    meta=pd.read_parquet(root/'data/samples.parquet'); assert meta.sample_id.is_unique
    src=project/'data/qualification/release_v1/sample_manifest.parquet';old=pd.read_parquet(src)
    assert old.sample_id.is_unique
    lookup=old.set_index('sample_id');joined=lookup.loc[meta.sample_id].reset_index()
    assert joined.sample_id.tolist()==meta.sample_id.tolist()
    assert (joined.teacher_sequence_sha256.to_numpy()==meta.sequence_sha256.to_numpy()).all()
    protected_groups=set(old.loc[old.role.isin(['physical_calibration','protected_historical']),'group_id'])
    assert not set(meta.group_id)&protected_groups
    # Missing/unresolved U is distinct from an observed T != U.
    known=joined.sequence.notna() & joined.sequence_sha256.notna()
    equal=known & (joined.teacher_sequence_sha256==joined.sequence_sha256)
    strata=pd.Series('U_unavailable',index=meta.index)
    strata[known]='T_differs_from_U';strata[equal]='strict_T_equals_U'
    mapping=dict(zip(meta.sample_id,strata))
    hs=project/'data/homology';plan=json.loads((hs/'search_plan.json').read_text());rec=json.loads((hs/'attempt_002/receipt.json').read_text())
    assert rec['status']=='complete' and sha(hs/'search_plan.json')==rec['plan_sha256']
    for name in ['new_U','old_views']:
        assert sha(hs/(name+'.fasta'))==plan['inputs'][name]['sha256']
    new= headers(hs/'new_U.fasta');olds=headers(hs/'old_views.fasta');T=set(meta.sequence_sha256)
    new_intersection=T&new;old_intersection=T&olds
    # Both stored directions include new_U as one endpoint. If no T is in
    # new_U, no stored alignment can compare one current T with another T.
    assert not new_intersection, 'Current T overlaps new_U: implement bounded direct alignment join instead'
    assert T<=olds
    scope=dict(status='not_estimable_from_available_pairwise_search',current_unique_T=len(T),
        T_in_new_U=len(new_intersection),T_in_old_views=len(old_intersection),
        stored_directions=plan['directions'],source_run=rec['run_id'],
        scope_proof='Both stored searches have new_U at one endpoint; zero current T sequences belong to new_U. Old-old pairwise relations are inherited as components, not available here as scored alignments.',
        interpretation='No reported nearest identity; unavailable is not identity zero and not evidence of no remote homology.',
        search_filters={k:plan[k] for k in ['minimum_identity','minimum_shorter_coverage','maximum_evalue','sensitivity']},
        required_next='Reuse the original old-old raw scored search if located, or separately authorize a bounded T-to-training similarity audit; no new search in this postprocessing.',
        source_files={str(hs/'search_plan.json'):sha(hs/'search_plan.json'),str(hs/'attempt_002/receipt.json'):sha(hs/'attempt_002/receipt.json')})
    selected_path=root/'results/selected_fits.csv';before=sha(selected_path);selected=pd.read_csv(selected_path)
    rows=[];audits=[];details=[]
    for fit in selected.to_dict('records'):
        run=Path(fit['artifact_path']);roles=json.loads((run/'roles.json').read_text())
        sets={k:set(v) for k,v in roles.items()}
        assert all(len(v)==len(sets[k]) for k,v in roles.items())
        m=meta.set_index('sample_id')
        gs={k:set(m.loc[list(v),'group_id']) for k,v in sets.items()}
        seq={k:set(m.loc[list(v),'sequence_sha256']) for k,v in sets.items()}
        for key in ['stop_ids','evaluation_ids']:
            assert not gs['train_ids']&gs[key] and not seq['train_ids']&seq[key]
        assert not gs['stop_ids']&gs['evaluation_ids']
        assert not gs['train_ids']&protected_groups
        frame=pd.read_parquet(run/'evaluation_metrics.parquet'); assert set(frame.sample_id)==sets['evaluation_ids']
        frame['stratum']=frame.sample_id.map(mapping);assert frame.stratum.notna().all()
        frame['run_id']=fit['run_id'];frame['family']=fit['family'];frame['scale']=str(fit['scale']);frame['fold']=fit['fold']
        frame['nearest_training_identity_status']='not_estimable_missing_T_to_T_scores'
        frame['nearest_training_identity']=float('nan');details.append(frame)
        for label,sub in frame.groupby('stratum'):
            rows.append(dict(run_id=fit['run_id'],family=fit['family'],scale=str(fit['scale']),fold=fit['fold'],stratum=label,**metrics(sub)))
        audits.append(dict(run_id=fit['run_id'],fold=fit['fold'],scale=str(fit['scale']),
            train_groups=len(gs['train_ids']),stop_groups=len(gs['stop_ids']),evaluation_groups=len(gs['evaluation_ids']),
            train_stop_group_overlap=0,train_eval_group_overlap=0,stop_eval_group_overlap=0,
            train_protected_group_overlap=0,train_eval_exact_T_overlap=0,
            roles_sha256=sha(run/'roles.json'),evaluation_metrics_sha256=sha(run/'evaluation_metrics.parquet')))
    assert sha(selected_path)==before,'selection changed during postprocessing'
    all_details=pd.concat(details,ignore_index=True);all_details.to_parquet(out/'selected_per_sample_strata.parquet',index=False)
    pd.DataFrame(rows).to_csv(out/'strict_T_U_per_fold.csv',index=False)
    pooled=[]
    for (family,scale,label),sub in all_details.groupby(['family','scale','stratum']):
        assert sub.sample_id.is_unique,'Each selected protein must be evaluated once across selected folds'
        pooled.append(dict(family=family,scale=scale,stratum=label,folds=sorted(sub.fold.unique().tolist()),**metrics(sub)))
    pd.DataFrame(pooled).to_csv(out/'strict_T_U_pooled.csv',index=False)
    paired=paired_comparisons(all_details[all_details.scale=='all'],'stratum')
    paired.to_csv(out/'paired_stratum_differences.csv',index=False)
    (out/'homology_distance_scope.json').write_text(json.dumps(scope,indent=2)+'\n')
    audit=dict(status='complete',selected_fits=len(selected),source_sample_manifest_sha256=sha(src),
        selected_fits_sha256=before,script_sha256=sha(__file__),dataset_stratum_counts=strata.value_counts().to_dict(),
        group_isolation_basis='Reused fullU historical successor T/U/C components; no fresh pairwise or profile search, no claim of all remote-homology exclusion',
        nearest_identity_status=scope['status'],fits=audits)
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    denominator=all_details[(all_details.family=='composition_charge')&(all_details.scale=='all')]
    total=dict(rows=len(denominator),unique_T=denominator.protein_id.nunique(),groups=denominator.group_id.nunique())
    counts=denominator.groupby('stratum').agg(rows=('sample_id','size'),unique_T=('protein_id','nunique'),groups=('group_id','nunique')).reset_index().to_dict('records')
    missing_ids=denominator.loc[denominator.stratum=='U_unavailable','sample_id']
    missing_reasons=lookup.loc[missing_ids,'binding_status'].value_counts(dropna=False).to_dict()
    den=dict(total=total,strata=counts,groups_spanning_multiple_strata=int((denominator.groupby('group_id').stratum.nunique()>1).sum()),
        proteins_spanning_multiple_strata=int((denominator.groupby('protein_id').stratum.nunique()>1).sum()),
        unknown_U_reason_counts=missing_reasons,
        note='Group counts are conditional within strata and are not additive. U_unavailable means no unique accepted parent U mapping, including absent, unsupported or ambiguous mapping; not a claim that no natural full sequence exists.')
    (out/'denominators.json').write_text(json.dumps(den,indent=2)+'\n')
    # Reuse the numerical boundaries in train_atlas_scale.py strata.json.
    # These marginal strata can be confounded with T/U relation and each other.
    extra=all_details[all_details.scale=='all'].copy()
    assert extra.length.notna().all() and (extra.length>0).all()
    assert extra.coverage.dropna().between(0,1).all()
    extra['length_bin']=pd.cut(extra.length,[0,150,300,600,np.inf],labels=['1-150','151-300','301-600','601+'],right=True).astype(str)
    extra['coverage_bin']=pd.cut(extra.coverage,[-np.inf,.7,.9,.98,np.inf],labels=['0-0.70','>0.70-0.90','>0.90-0.98','>0.98-1.00'],right=True).astype('object').fillna('unavailable')
    heterogeneity=[];heterogeneity_paired=[];cross_counts=[]
    for axis in ['length_bin','coverage_bin']:
        for (family,label),sub in extra.groupby(['family',axis],observed=True):
            heterogeneity.append(dict(axis=axis,stratum=str(label),family=family,scale='all',**metrics(sub)))
        pair=paired_comparisons(extra,axis);pair.insert(0,'axis',axis);heterogeneity_paired.append(pair)
        denom=extra[extra.family=='composition_charge']
        for (label,relation),sub in denom.groupby([axis,'stratum'],observed=True):
            cross_counts.append(dict(axis=axis,bin=str(label),T_U_relation=relation,rows=len(sub),unique_T=sub.protein_id.nunique(),groups=sub.group_id.nunique()))
    heterogeneity=pd.DataFrame(heterogeneity);heterogeneity_paired=pd.concat(heterogeneity_paired,ignore_index=True)
    heterogeneity.to_csv(out/'length_coverage_pooled.csv',index=False)
    heterogeneity_paired.to_csv(out/'length_coverage_paired.csv',index=False)
    pd.DataFrame(cross_counts).to_csv(out/'length_coverage_relation_counts.csv',index=False)
    definitions=dict(length='Length of model input teacher sequence T, in amino acids; bins (0,150], (150,300], (300,600], (600,infinity).',
        coverage='Existing data/samples coverage field inherited from T_U_coverage; bins [0,.7], (.7,.9], (.9,.98], (.98,1]; unavailable separate. Same boundaries as training strata.json.',
        metric='Sample H is averaged within group/protein, then proteins within group, then groups equally within each conditional stratum.',
        limitation='Marginal retrospective strata; group counts overlap across strata and are not additive; T/U relation, length and coverage are correlated; no causal heterogeneity or independence claim.',
        bootstrap='2000 paired group resamples seed3407; post-selection exploratory 95% intervals, no selection or multiplicity correction; no seed variation.',
        folds=[0,1],selected_fits=len(selected))
    (out/'strata_definitions.json').write_text(json.dumps(definitions,indent=2)+'\n')
    public=out/'public_aggregate';public.mkdir(exist_ok=True)
    aggregate_tables={'strict_T_U_per_fold.csv':pd.DataFrame(rows).drop(columns=['run_id']),
        'strict_T_U_pooled.csv':pd.DataFrame(pooled),'paired_stratum_differences.csv':paired,
        'length_coverage_pooled.csv':heterogeneity,'length_coverage_paired.csv':heterogeneity_paired,
        'length_coverage_relation_counts.csv':pd.DataFrame(cross_counts)}
    forbidden={'sample_id','protein_id','group_id','run_id','artifact_path','sequence','sequence_sha256'}
    for name,table in aggregate_tables.items():
        assert not forbidden&set(table.columns)
        table.to_csv(public/name,index=False)
    for name,value in [('denominators.json',den),('strata_definitions.json',definitions),
        ('homology_distance_scope.json',{k:v for k,v in scope.items() if k not in ['source_files','source_run','stored_directions']})]:
        (public/name).write_text(json.dumps(value,indent=2)+'\n')
    public_audit=dict(status='complete',selected_fits=len(selected),folds=[0,1],
        selected_fits_sha256=before,script_sha256=sha(__file__),protected_group_overlaps=0,
        train_stop_group_overlaps=0,train_eval_group_overlaps=0,stop_eval_group_overlaps=0,
        train_eval_exact_T_overlaps=0,contents='Aggregate metrics and counts only; no sample/group/protein/run identifiers, sequences, predictions or checkpoint files.',
        qualification='Retrospective development comparison, not blind confirmation. Both component isolation and exact-T isolation checked for each selected fit.')
    (public/'audit.json').write_text(json.dumps(public_audit,indent=2)+'\n')
    text=['# T/U分层分母与解释','',f"共同评价集：{total['rows']}行，{total['unique_T']}唯一T，{total['groups']}同源组。",'',
        '| 分层 | 行数 | 唯一T | 组数 |','|---|---:|---:|---:|']
    for v in counts:text.append(f"| {v['stratum']} | {v['rows']} | {v['unique_T']} | {v['groups']} |")
    text.extend(['',f"{den['groups_spanning_multiple_strata']}个同源组横跨多个分层，分层组数不能直接相加；行数完整覆盖共同评价集。",'',
        'U_unavailable表示没有唯一、获支持的U关联；它与已明确T≠U分开。具体原因见denominators.json。',
        'strict_T_equals_U仅指序列hash相等，不自动等同旧1061条完整资格U队列，也不单独证明天然全长部署可行。',
        '配对bootstrap是选型后的探索性描述，无多模型选择校正；区间跨0不能解释为等效。',
        'length_coverage_*是全训练规模的边际分层；长度、覆盖、T/U关系彼此相关，不作独立或因果异质性解释。',
        '现有两方向同源搜索均含new_U端点，本轮T在new_U集合交集为0，因此无法从这些文件给出当前T到训练T的直接identity；未检出/缺少搜索不填0。'])
    (out/'README.md').write_text('\n'.join(text)+'\n')
    (public/'README.md').write_text('\n'.join(text)+'\n')
    assert sha(selected_path)==before,'selection changed during postprocessing'
    print(json.dumps({k:v for k,v in audit.items() if k!='fits'}));print(pd.DataFrame(pooled).to_string(index=False))

if __name__=='__main__':main()
