"""Private, offline OOF diagnostics; no fitting of predictive models or GPU use."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,hashlib,html,json,resource,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr,pearsonr,rankdata

FAMILIES=['composition_charge_deep','mean_pre','residue_mean']
SHORT={'composition_charge_deep':'Charge MLP','mean_pre':'Mean MLP','residue_mean':'Residue mean'}
COLORS={'composition_charge_deep':'#737b86','mean_pre':'#2673b8','residue_mean':'#dc6b2f'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str,allow_nan=False)+'\n',encoding='utf8')
def groupmeans(d):return d.groupby(['group_id','protein_id'])[['H','CDF','tail']].mean().groupby('group_id').mean()
def measures(d):
 g=groupmeans(d)
 return dict(rows=len(d),unique_T=d.protein_id.nunique(),groups=len(g),**{k:float(v) for k,v in g.mean().items()})
def errorframe(y,p,meta):
 assert p.shape==y.shape and np.isfinite(p).all() and (p>=0).all() and abs(p.sum(1)-1).max()<1e-6
 d=meta.copy();d['H']=np.sqrt(.5*((np.sqrt(y.astype('float64'))-np.sqrt(p.astype('float64')))**2).sum(1))
 d['CDF']=((y.astype('float64').cumsum(1)[:,:-1]-p.astype('float64').cumsum(1)[:,:-1])**2).mean(1)
 d['tail']=abs(y[:,[0,-1]].astype('float64')-p[:,[0,-1]].astype('float64')).mean(1)
 return d
def savefig(fig,folder,name):
 fig.savefig(folder/(name+'.png'),dpi=160,bbox_inches='tight');fig.savefig(folder/(name+'.svg'),bbox_inches='tight');plt.close(fig)
def table(d):return d.to_html(index=False,border=0,classes='data',float_format=lambda x:f'{x:.6g}',escape=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--project',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
 resource.setrlimit(resource.RLIMIT_AS,(16<<30,16<<30));start=time.monotonic();project=Path(a.project);root=Path(a.output)
 for name in ['private','public','plots','cases']: (root/name).mkdir(parents=True,exist_ok=True)
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})
 atlas=project/'atlas_scale_20261105';night=project/'overnight_20261106';fitfile=night/'confirmation/results/all_fits.csv';fits=pd.read_csv(fitfile);assert len(fits)==30
 meta=pd.read_parquet(atlas/'data/samples.parquet');y=np.load(atlas/'data/p.npy');release=json.loads((atlas/'data/release.json').read_text())
 for name in ['samples.parquet','p.npy']:assert sha(atlas/'data'/name)==release['artifacts'][name]
 oldpath=project/'data/qualification/release_v1/sample_manifest.parquet';old=pd.read_parquet(oldpath).set_index('sample_id').loc[meta.sample_id].reset_index()
 assert (old.teacher_sequence_sha256.to_numpy()==meta.sequence_sha256.to_numpy()).all()
 meta['relation']=np.where(old.sequence_sha256.isna(),'U unavailable',np.where(old.sequence_sha256==old.teacher_sequence_sha256,'T = U','T differs U'))
 meta['parent_origin']=meta.source_origin.fillna('unavailable');meta['cohort_group_T']=meta.group_id.map(meta.groupby('group_id').protein_id.nunique())
 meta['length_bin']=pd.cut(meta.length,[0,150,300,600,np.inf],labels=['<=150','151-300','301-600','>600']).astype(str)
 meta['coverage_bin']=pd.cut(meta.coverage,[-np.inf,.7,.9,.98,np.inf],labels=['<=.70','(.70,.90]','(.90,.98]','>.98']).astype('object').fillna('unavailable')
 meta['group_size_bin']=pd.cut(meta.cohort_group_T,[0,1,5,20,np.inf],labels=['1','2-5','6-20','>20']).astype(str)
 ix=pd.Series(np.arange(len(meta)),index=meta.sample_id);predictions={};details=[];curves=[];provenance=[];fit_lookup={}
 for fit in fits.to_dict('records'):
  run=Path(fit['artifact_path']);rec=json.loads((run/'receipt.json').read_text());roles=json.loads((run/'roles.json').read_text());ids=ix.loc[roles['evaluation_ids']].to_numpy()
  p=np.load(run/'evaluation_prediction.npy');assert sha(run/'evaluation_prediction.npy')==rec['prediction_sha256']
  assert (meta.iloc[ids].fold==fit['fold']).all();g={k:set(meta.iloc[ix.loc[v].to_numpy()].group_id) for k,v in roles.items()};assert not(g['train_ids']&g['evaluation_ids'] or g['train_ids']&g['stop_ids'] or g['stop_ids']&g['evaluation_ids'])
  key=(fit['family'],fit['seed']);predictions.setdefault(key,np.full(y.shape,np.nan,dtype='float32'))[ids]=p
  d=errorframe(y[ids],p,meta.iloc[ids].copy());assert abs(measures(d)['H']-fit['H'])<1e-7
  d['family']=fit['family'];d['seed']=fit['seed'];details.append(d)
  curve=pd.DataFrame(json.loads((run/'curve.json').read_text()));curve['family']=fit['family'];curve['fold']=fit['fold'];curve['seed']=fit['seed'];curves.append(curve)
  fit_lookup[(fit['family'],fit['seed'],fit['fold'])]=dict(run=run,roles=roles,rec=rec)
  provenance.append(dict(family=fit['family'],fold=fit['fold'],seed=fit['seed'],prediction_sha256=sha(run/'evaluation_prediction.npy'),curve_sha256=sha(run/'curve.json'),roles_sha256=sha(run/'roles.json')))
 for key,p in predictions.items():assert np.isfinite(p).all()
 allrows=pd.concat(details,ignore_index=True);allrows.drop(columns=['sequence']).to_parquet(root/'private/oof_errors.parquet',index=False)
 cframe=pd.concat(curves,ignore_index=True);cframe.to_csv(root/'private/training_curves.csv',index=False)
 scores=[];strata=[]
 for (family,seed),d in allrows.groupby(['family','seed']):
  assert d.sample_id.is_unique and len(d)==len(meta);scores.append(dict(family=family,seed=seed,**measures(d)))
  for axis in ['length_bin','coverage_bin','relation','group_size_bin','parent_origin']:
   for label,sub in d.groupby(axis):strata.append(dict(family=family,seed=seed,axis=axis,stratum=str(label),**measures(sub)))
 scores=pd.DataFrame(scores);strata=pd.DataFrame(strata);scores.to_csv(root/'public/scores.csv',index=False);strata.to_csv(root/'public/strata.csv',index=False)
 # Training and stop metrics have different weighting and units; no gap claim.
 for seed in [1701,3407]:
  fig,axes=plt.subplots(3,2,figsize=(12,10))
  for row,family in enumerate(FAMILIES):
   for fold in range(5):
    cc=cframe[(cframe.family==family)&(cframe.seed==seed)&(cframe.fold==fold)];color=plt.cm.tab10(fold)
    axes[row,0].plot(cc.epoch,cc.train_H2,label=f'fold {fold}',color=color);axes[row,1].plot(cc.epoch,cc.stop_H,label=f'fold {fold}',color=color)
    best=fit_lookup[(family,seed,fold)]['rec']['best_epoch'];value=cc.loc[cc.epoch==best,'stop_H']
    if len(value):axes[row,1].scatter([best],[value.iloc[0]],s=24,color=color)
   axes[row,0].set(yscale='log',ylabel='Training H-squared loss',title=SHORT[family]+' / training')
   axes[row,1].set(ylim=(.04,.10),ylabel='Stop group-macro H',title=SHORT[family]+' / stop (dots: chosen epoch)')
   for ax in axes[row]:ax.set_xlabel('Epoch');ax.grid(alpha=.2)
  axes[0,1].legend(ncol=5,fontsize=8);fig.suptitle(f'Training vs stop | seed {seed} | units/weights differ; stop panel zoomed')
  fig.tight_layout();savefig(fig,root/'plots',f'curves_{seed}')
 # Continuation panels retain three separate configurations, not pooled OOF.
 explore=night/'explore';es=json.loads((explore/'results/summary.json').read_text());cont=[Path(es['selected_by_stop']['mean_continue']['checkpoint']).parent]+[p.parent for p in sorted((explore/'confirmation/runs').glob('*/receipt.json'))]
 assert len(cont)==3;fig,axes=plt.subplots(1,3,figsize=(14,3.7));continuation=[]
 for ax,run in zip(axes,cont):
  rec=json.loads((run/'receipt.json').read_text());cfg=rec['config'];fold=cfg['fold'];seed=cfg['seed'];cc=pd.DataFrame(json.loads((run/'curve.json').read_text()));source=Path(cfg['warmstart_path']).parent;sr=json.loads((source/'receipt.json').read_text());sc=pd.DataFrame(json.loads((source/'curve.json').read_text()))
  ax.plot(sc.epoch,sc.stop_H,color='#2673b8',label='Original stop H');offset=int(sr['best_epoch']);ax.plot(offset+np.r_[0,cc.epoch],np.r_[rec['initial_stop_H'],cc.stop_H],color='#dc6b2f',label='Continue from best')
  ax.axvline(offset,color='#999',ls=':',lw=1);ax.set(title=f'Fold {fold}, seed {seed}',xlabel='Epoch (continuation offset = source best)',ylabel='Stop H',ylim=(.045,.06));ax.grid(alpha=.2)
  continuation.append(dict(fold=fold,seed=seed,source_best_epoch=offset,continued_best_epoch=rec['best_epoch'],source_H=sr['result']['group_macro_H'],continued_H=rec['result']['group_macro_H'],source_stop=rec['initial_stop_H'],continued_stop=rec['best_stop_H']))
 axes[0].legend(fontsize=8);fig.suptitle('Three continuation diagnostics: restarting optimizer + lower LR, not a complete grid');fig.tight_layout();savefig(fig,root/'plots','continuation');dump(root/'public/continuation.json',continuation)
 # Conditional group macro metrics; subgroup groups overlap, counts not additive.
 for axis in ['length_bin','coverage_bin','relation','group_size_bin','parent_origin']:
  fig,axes=plt.subplots(1,3,figsize=(14,4.4));ss=strata[(strata.axis==axis)&(strata.seed==1701)];labels=ss.stratum.unique().tolist()
  preferred={'length_bin':['<=150','151-300','301-600','>600'],'coverage_bin':['<=.70','(.70,.90]','(.90,.98]','>.98','unavailable'],'group_size_bin':['1','2-5','6-20','>20']}
  if axis in preferred:labels=[v for v in preferred[axis] if v in labels]
  for ax,metric in zip(axes,['H','CDF','tail']):
   for j,fam in enumerate(FAMILIES):
    z=ss[ss.family==fam].set_index('stratum').loc[labels];ax.plot(np.arange(len(labels))+(j-1)*.07,z[metric],marker='o',ls='none' if axis in ['parent_origin','relation'] else '-',label=SHORT[fam],color=COLORS[fam])
   ax.set_xticks(range(len(labels)),labels,rotation=28,ha='right');ax.set_ylabel('Group macro '+metric);ax.grid(alpha=.2)
  sizes=ss[ss.family=='residue_mean'].set_index('stratum').loc[labels]
  axes[0].set_xticklabels([f'{l}\nn={int(n)}, g={int(g)}' for l,n,g in zip(labels,sizes.rows,sizes.groups)],rotation=28,ha='right');axes[0].legend(fontsize=8)
  fig.suptitle(f'{axis} | seed1701 | conditional group averages; see both seeds in tables');fig.tight_layout();savefig(fig,root/'plots','strata_'+axis)
 # Correlations at unique-T level. Grouped CV fold range and partial checks
 # expose sensitivity; coefficients are descriptive, no IID significance claim.
 correlations=[]
 for (family,seed),d in allrows.groupby(['family','seed']):
  agg={'H':'mean','CDF':'mean','tail':'mean','length':'first','coverage':'first','cohort_group_T':'first','fold':'first','group_id':'first','relation':'first','parent_origin':'first'}
  u=d.groupby('protein_id').agg(agg).reset_index();u['log10_length']=np.log10(u.length);u['log1p_group_T']=np.log1p(u.cohort_group_T)
  for predictor in ['log10_length','coverage','log1p_group_T']:
   sub=u[u[predictor].notna()].copy();xx=sub[predictor].to_numpy();cov=sub[['log10_length','coverage','log1p_group_T']].drop(columns=[predictor]);cov['coverage_missing']=cov.get('coverage',pd.Series(0,index=cov.index)).isna().astype(float)
   cov=cov.fillna(0);cat=pd.get_dummies(sub[['relation','parent_origin']].astype(str),drop_first=True,dtype=float);C=np.column_stack([np.ones(len(sub)),cov.to_numpy(),cat.to_numpy()]);rankC=C.copy();rankC[:,1:1+len(cov.columns)]=np.column_stack([rankdata(cov[c]) for c in cov.columns])
   for metric in ['H','CDF','tail']:
    yy=np.log10(sub[metric].to_numpy()+1e-10);raws=float(spearmanr(xx,yy).statistic);rawp=float(pearsonr(xx,yy).statistic)
    rx=xx-C@np.linalg.lstsq(C,xx,rcond=None)[0];ry=yy-C@np.linalg.lstsq(C,yy,rcond=None)[0]
    rrx=rankdata(xx)-rankC@np.linalg.lstsq(rankC,rankdata(xx),rcond=None)[0];rry=rankdata(yy)-rankC@np.linalg.lstsq(rankC,rankdata(yy),rcond=None)[0]
    foldrho=[float(spearmanr(v[predictor],v[metric]).statistic) for _,v in sub.groupby('fold')]
    correlations.append(dict(family=family,seed=seed,metric=metric,predictor=predictor,n_T=len(sub),groups=sub.group_id.nunique(),spearman=raws,pearson_log_error=rawp,partial_pearson=float(pearsonr(rx,ry).statistic),partial_rank=float(pearsonr(rrx,rry).statistic),fold_spearman_min=min(foldrho),fold_spearman_max=max(foldrho),control_rank=int(np.linalg.matrix_rank(C))))
 correlations=pd.DataFrame(correlations);correlations.to_csv(root/'public/correlations.csv',index=False)
 fig,axes=plt.subplots(1,2,figsize=(13,5.6))
 for ax,seed in zip(axes,[1701,3407]):
  c=correlations[(correlations.family=='residue_mean')&(correlations.seed==seed)].copy();c['row']=c.metric+' / '+c.predictor;cols=['spearman','pearson_log_error','partial_pearson','partial_rank'];v=c[cols].to_numpy();im=ax.imshow(v,vmin=-.5,vmax=.5,cmap='coolwarm',aspect='auto');ax.set_xticks(range(4),['Spearman','Pearson log(err)','Partial Pearson','Partial rank'],rotation=25,ha='right');ax.set_yticks(range(len(c)),c.row);ax.set_title(f'Residue mean / seed {seed}')
  for i in range(len(c)):
   for j in range(4):ax.text(j,i,f'{v[i,j]:.2f}',ha='center',va='center',fontsize=8)
 fig.suptitle('Error correlations | unique T, correlated groups; descriptive only');fig.tight_layout();savefig(fig,root/'plots','correlations')
 d=allrows[(allrows.family=='residue_mean')&(allrows.seed==1701)].drop_duplicates('protein_id');fig,axes=plt.subplots(1,3,figsize=(14,4))
 for ax,col,label in zip(axes,['length','coverage','cohort_group_T'],['Teacher length (log10)','T/U coverage','Cohort group unique T (log1p)']):
  v=d[col];x=np.log10(v) if col=='length' else np.log1p(v) if col=='cohort_group_T' else v;ok=x.notna();ax.hexbin(x[ok],d.loc[ok,'H'],gridsize=40,mincnt=1,bins='log',cmap='viridis');ax.set(xlabel=label,ylabel='Per-row H',title=f'n={int(ok.sum())} unique T');ax.grid(alpha=.15)
 fig.suptitle('Descriptive density | residue mean, seed1701 | no causal interpretation');fig.tight_layout();savefig(fig,root/'plots','error_density')
 # Fixed-order deterministic case rules on canonical sample per unique T.
 base=errorframe(y,predictions[('residue_mean',1701)],meta.copy());charge=errorframe(y,predictions[('composition_charge_deep',1701)],meta.copy());base['delta_charge']=base.H-charge.H
 eligible=base.sort_values('sample_id').drop_duplicates('protein_id');median=float(eligible.H.median());eligible=eligible.copy();eligible['typical_distance']=abs(eligible.H-median)
 used=set();used_groups=set();selected=[]
 for kind,col,ascending in [('best','H',True),('worst','H',False),('typical','typical_distance',True),('improved_vs_charge','delta_charge',True),('regressed_vs_charge','delta_charge',False)]:
  pool=eligible[(~eligible.protein_id.isin(used))&(~eligible.group_id.isin(used_groups))].sort_values([col,'sample_id'],ascending=[ascending,True]).drop_duplicates('group_id').head(2)
  for _,v in pool.iterrows():used.add(v.protein_id);used_groups.add(v.group_id);selected.append((kind,int(v.name)))
 assert len(selected)==10
 # Case feature-neighbor proxy uses only the same fit's train fold; it is not
 # sequence identity, homology or support of an electrostatic target.
 chargefeatures=np.load(atlas/'data/composition_charge.npy');cache=atlas/'data/esmc_300m_mean';ei=pd.read_parquet(cache/'index.parquet');emb=np.load(cache/'mean.npy');ep=pd.Series(np.arange(len(ei)),index=ei.sequence_sha256).loc[meta.sequence_sha256].to_numpy();emb=emb[ep];emb=emb/np.linalg.norm(emb,axis=1,keepdims=True)
 # Historical p uses its own fixed cutpoints, not the newer C protocol bins.
 label_binding_path=atlas/'data/label_contract_addendum.json';label_binding=json.loads(label_binding_path.read_text())
 assert label_binding['release_sha256']==sha(atlas/'data/release.json')
 assert sha(label_binding['label_contract_path'])==label_binding['label_contract_sha256']
 edges=np.asarray(label_binding['label_contract']['edges'],dtype=float)
 assert len(edges)==62 and np.isfinite(edges).all() and (np.diff(edges)>0).all()
 assert label_binding['label_contract']['phase']=='historical' and label_binding['label_contract']['units']=='kBT/e'
 cases=[];case_sections=[]
 for number,(kind,i) in enumerate(selected,1):
  row=meta.iloc[i];oldrow=old.iloc[i];fold=int(row.fold);fit=fit_lookup[('residue_mean',1701,fold)];tr=ix.loc[fit['roles']['train_ids']].to_numpy();tr=meta.iloc[tr].drop_duplicates('protein_id').index.to_numpy()
  raw=chargefeatures;med=np.nan_to_num(np.nanmedian(raw[tr],axis=0));filled=np.where(np.isnan(raw),med,raw);mu=filled[tr].mean(0);sd=filled[tr].std(0);sd[sd<1e-6]=1
  z=(filled-mu)/sd;dist=np.sqrt(((z[tr]-z[i])**2).mean(1));nearest=int(tr[np.argmin(dist)]);similarity=emb[tr]@emb[i];near_e=int(tr[np.argmax(similarity)])
  samegroup=int((meta.iloc[tr].group_id==row.group_id).sum());assert samegroup==0
  info=dict(case=number,rule=kind,sample_id=row.sample_id,protein_id=row.protein_id,group_id=row.group_id,fold=fold,length=int(row.length),U_length=None if pd.isna(oldrow.U_length) else int(oldrow.U_length),relation=row.relation,coverage=None if pd.isna(row.coverage) else float(row.coverage),binding_status=str(oldrow.binding_status),parent_origin=row.parent_origin,parent_source_id=str(oldrow.source_sequence_id),parent_release=str(oldrow.source_release),structure_source=str(row.source),structure_sha256=str(row.source_structure_sha256),cohort_group_T=int(row.cohort_group_T),same_group_training_T=samegroup,nearest_charge_RMS=float(dist.min()),nearest_charge_training_id=str(meta.iloc[nearest].sample_id),nearest_embedding_cosine=float(similarity.max()),nearest_embedding_training_id=str(meta.iloc[near_e].sample_id),H=float(base.iloc[i].H),delta_H_vs_charge=float(base.iloc[i].delta_charge))
  ps={'True p':y[i],**{SHORT[fam]:predictions[(fam,1701)][i] for fam in FAMILIES}}
  fig,axes=plt.subplots(1,2,figsize=(12,3.9))
  for name,p in ps.items():
   color='black' if name=='True p' else COLORS[[k for k in FAMILIES if SHORT[k]==name][0]]
   axes[0].plot(np.arange(63),p,label=name,color=color,lw=2 if name=='True p' else 1.3);axes[1].step(edges,p.cumsum()[:-1],where='post',label=name,color=color,lw=2 if name=='True p' else 1.3)
  axes[0].set(xlabel='Bin index (0 / 62 = open tail bins)',ylabel='Probability mass');axes[1].set(xlabel='Potential cutpoint (kBT/e)',ylabel='CDF at 62 finite cutpoints',ylim=(-.02,1.02));axes[0].legend(fontsize=8)
  for ax in axes:ax.grid(alpha=.2)
  fig.suptitle(f'Case {number:02d} | {kind} | {row.sample_id} | T={row.length} aa | fold {fold}');fig.tight_layout();savefig(fig,root/'cases',f'case_{number:02d}')
  # Probability arrays and sequences remain explicitly private.
  dump(root/'private'/f'case_{number:02d}.json',dict(metadata=info,teacher_sequence=row.sequence,parent_sequence=oldrow.sequence,truth_p=y[i].tolist(),predicted_p={k:v.tolist() for k,v in ps.items() if k!='True p'}))
  metricrows=[]
  for fam in FAMILIES:
   mm=errorframe(y[[i]],predictions[(fam,1701)][[i]],meta.iloc[[i]]);metricrows.append(dict(model=SHORT[fam],**{k:float(mm.iloc[0][k]) for k in ['H','CDF','tail']}))
  cases.append(info);case_sections.append(f'<section id="case{number}"><h2>案例 {number:02d} · {html.escape(kind)} · {html.escape(str(row.sample_id))}</h2><img src="cases/case_{number:02d}.png" alt="真值与三模型分布及CDF"><p>{html.escape(json.dumps(info,ensure_ascii=False))}</p>{table(pd.DataFrame(metricrows))}<details><summary>私有T序列与U关联</summary><pre>T: {html.escape(str(row.sequence))}\nU: {html.escape(str(oldrow.sequence))}</pre></details><p>同组训练T为0由划分保证。近邻为训练折中的特征空间proxy，不是同源identity，也不能证明物理适用域。结构坐标未在本诊断输入合同中绑定，因此没有生成猜测性结构图。</p></section>')
 dump(root/'private/case_selection.json',dict(seed=1701,order=['best','worst','typical','improved_vs_charge','regressed_vs_charge'],per_rule=2,deduplicate='one lexicographically first sample per unique T; at most one T per homology group across all rules',revision_note='First render audit found two improved examples from one group; deterministic group deduplication added for review diversity, no manual replacement by visual impression.',typical='closest H to median across canonical T rows',cases=cases))
 # Root causes are hypotheses grounded in matching controls, never diagnoses.
 narrative='''<h2>如何解读：原因候选与替代解释</h2><ul><li><b>优化仍有余量：</b>三次低LR续训的H/CDF/尾误差均改善，支持继续训练预算是一个真实因素。它同时重启AdamW并增加更新，不能把全部收益单独归因于LR，也未形成完整五折两seed。</li><li><b>mean前信息压缩：</b>同参数、共同输入的残基非线性后聚合总体优于先mean，支持当前读出可利用mean先行丢失的信号；优化轨迹、正则化和样本权重仍是替代解释。该比较没有证明序列顺序无用。</li><li><b>输入/标签对应：</b>T≠U、缺失U及低覆盖是不同问题。标签仍是历史T结构的顶点p，而非完整U真值；coverage相关不能被解释为模型在全长部署的准确性。source_origin描述亲本序列关联来源，不是结构实验方法。</li><li><b>组/来源支持不足：</b>所有物理标签来自同一PDB历史来源，无法从本队列辨识PDB与AF跨来源效应；组大小与难度的相关也可能由家族、长度或折分布造成。特征近邻不是序列搜索。</li><li><b>极端案例：</b>固定最差/退化规则用于定位需要人工核查的样本；分布尾部、构建体、多段连接及标签来源值得核查，但没有重新计算物理或验证坐标，不能把个例错误直接归因于坏标签。</li></ul>'''
 css='body{font:16px/1.6 system-ui,sans-serif;color:#172331;background:#f4f6f8;margin:0}main{max-width:1260px;margin:auto;padding:30px}section,.box{background:white;padding:24px;margin:20px 0;border-radius:12px;box-shadow:0 2px 12px #1231}h1,h2{line-height:1.25}nav{position:sticky;top:0;background:#17364d;color:white;padding:12px;z-index:2}nav a{color:white;margin-right:16px}a{color:#175f90}img{max-width:100%;height:auto}table{border-collapse:collapse;font-size:13px;display:block;overflow:auto}td,th{padding:7px;border-bottom:1px solid #dde4eb;text-align:right}th{background:#eaf1f6}td:first-child,th:first-child{text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}p{overflow-wrap:anywhere}.warning{border-left:5px solid #d57a24;padding-left:15px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}@media(max-width:850px){.grid{display:block}main{padding:12px}}'
 headings=''.join(f'<a href="#case{i}">{i:02d}</a>' for i in range(1,11))
 doc=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Atlas 私有误差审阅</title><style>{css}</style><body><nav><a href="#overview">总览</a><a href="#curves">训练</a><a href="#strata">分层</a><a href="#correlation">相关</a>案例 {headings}</nav><main><h1>Atlas surrogate · 私有人工误差审阅</h1><p>2026-11-06 · 30配置完整OOF + 3次续训 · 21,921行 / 21,915 T / 7,999组 · 无新训练或GPU</p><p class="warning">本页包含私有样本ID、序列、结构来源及案例曲线。仅留本地/Byte，禁止进入公开Git。全部五折已曝光，属于开发分析；两个seed分开报告。</p><section id="overview"><h2>总览与口径</h2>{table(scores)}<p>逐行计算H、62个CDF切点MSE和首末bin质量MAE，再同T内、组内T、组间依次平均。不是先平均p再计算H。子群同源组可以跨多个类别，组数不能相加。T=U只表示当前hash相等。</p>{narrative}</section><section id="curves"><h2>训练与stop曲线</h2><p>左栏为训练H² loss，右栏为stop组宏H，单位、dropout和样本权重不同，不能据两条数值差直接量化过拟合。右图缩放到0.04–0.10，完整数值在private/training_curves.csv。</p><img src="plots/curves_1701.png"><img src="plots/curves_3407.png"><img src="plots/continuation.png">{table(pd.DataFrame(continuation))}</section><section id="strata"><h2>分层与覆盖</h2><p>图展示seed1701；两seed的完整分层表保留于下方。组大小是整个合格开发队列中的唯一T数，不是训练折同组支持。</p>'''+''.join(f'<img src="plots/strata_{axis}.png">' for axis in ['length_bin','coverage_bin','relation','group_size_bin','parent_origin'])+f'''<details><summary>展开完整双seed分层表</summary>{table(strata)}</details></section><section id="correlation"><h2>误差相关：不把一个系数当结论</h2><p>每个唯一T一条，保留组数和五折Spearman范围。Pearson使用log10(error+1e−10)，长度log10、组大小log1p；Spearman对相同变量取秩。Partial控制其他连续量、coverage缺失标记、T/U关系与亲本关联来源虚拟变量。物理来源全是PDB，无法控制或估计PDB/AF差异；共线类别由最小二乘处理，控制矩阵秩单列。没有IID p值或因果断言。</p><img src="plots/correlations.png"><img src="plots/error_density.png"><details><summary>展开全部系数、样本数及折间范围</summary>{table(correlations)}</details></section><section><h2>案例选择规则（先定义后取样）</h2><p>固定seed1701，对每个唯一T取字典序最早样本；分组去重在首轮渲染核验发现重复组后加入，未按图形好坏手动替换。按best、worst、典型、相对charge改善最大、退化最大顺序各取2条，跨类同时按唯一T和同源组去重。best/worst用residue H，典型为距全体中位H最近，改善/退化用residue−charge H。极端案例不代表发生率；不按图形观感手工挑选。</p>{table(pd.DataFrame(cases)[['case','rule','sample_id','length','relation','coverage','cohort_group_T','H','delta_H_vs_charge']])}</section>'''+''.join(case_sections)+'''<section><h2>文件与边界</h2><p>private/保存OOF误差、训练曲线、案例序列和数组；public/仅无样本明细聚合；plots/为匿名静态科学图，cases/包含私有标识。没有网络请求、坐标下载、新同源搜索、模型训练或物理计算。</p></section></main></body></html>'''
 (root/'index.html').write_text(doc,encoding='utf8')
 # Static integrity QA before human visual inspection.
 from html.parser import HTMLParser
 class Assets(HTMLParser):
  def handle_starttag(self,tag,attrs):
   for k,v in attrs:
    if k=='src':assert not v.startswith(('http:','https:','//')) and (root/v).is_file(),v
 Assets().feed(doc)
 assert len(list((root/'cases').glob('*.png')))==10
 from PIL import Image
 for p in list((root/'plots').glob('*.png'))+list((root/'cases').glob('*.png')):
  im=Image.open(p);im.verify()
 receipt=dict(status='complete',OOF_configs=30,continuations=3,rows=21921,unique_T=21915,groups=7999,cases=10,CPU_threads=4,GPU=False,elapsed_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
  script_sha256=sha(__file__),historical_label_contract_sha256=label_binding['label_contract_sha256'],historical_label_binding_sha256=sha(label_binding_path),CDF_axis_cutpoint_range=[float(edges[0]),float(edges[-1])],CDF_axis_units='kBT/e',fit_index_sha256=sha(fitfile),source_manifest_sha256=sha(oldpath),source_release_sha256=sha(atlas/'data/release.json'),environment=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__),case_selection='fixed deterministic five rules, two cases each, unique T and homology group deduplicated',external_requests=0,
  checks=dict(probabilities_valid=True,OOF_complete=True,groups_disjoint=True,raw_prediction_metrics_match=True,all_HTML_images_exist=True,PNG_integrity_pass=True),source_fits=provenance)
 dump(root/'public/receipt.json',receipt);print(json.dumps({k:v for k,v in receipt.items() if k not in ['source_fits','environment']}))
if __name__=='__main__':main()
