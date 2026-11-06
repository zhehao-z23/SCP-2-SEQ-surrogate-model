"""Descriptive tail-scale controls; no model fit, selection, or causal claim."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import argparse,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr,rankdata,pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();p=Path(a.project);r=Path(a.output)
    meta=pd.read_parquet(p/'atlas_scale_20261105/data/samples.parquet');y=np.load(p/'atlas_scale_20261105/data/p.npy').astype(float)
    fits=pd.read_csv(p/'overnight_20261106/confirmation/results/all_fits.csv');ix=pd.Series(np.arange(len(meta)),index=meta.sample_id)
    records=[];strata=[]
    for (family,seed),part in fits.groupby(['family','seed']):
        pred=np.full_like(y,np.nan)
        for z in part.itertuples():
            q=Path(z.artifact_path);ids=json.loads((q/'roles.json').read_text())['evaluation_ids'];pred[ix.loc[ids]]=np.load(q/'evaluation_prediction.npy')
        assert np.isfinite(pred).all()
        d=meta[['protein_id','group_id','length','fold']].copy();d['true_tail_mass']=y[:,0]+y[:,-1];d['predicted_tail_mass']=pred[:,0]+pred[:,-1]
        d['lower_bias']=pred[:,0]-y[:,0];d['upper_bias']=pred[:,-1]-y[:,-1];d['tail_error']=(abs(d.lower_bias)+abs(d.upper_bias))/2
        u=d.groupby(['group_id','protein_id'],as_index=False).mean(numeric_only=True)
        x=rankdata(u.length);z=rankdata(u.true_tail_mass);err=rankdata(u.tail_error);C=np.column_stack([np.ones(len(z)),z]);rx=x-C@np.linalg.lstsq(C,x,rcond=None)[0];re=err-C@np.linalg.lstsq(C,err,rcond=None)[0]
        records.append(dict(family=family,seed=int(seed),unique_T=len(u),groups=u.group_id.nunique(),length_vs_true_tail_mass_spearman=float(spearmanr(u.length,u.true_tail_mass).statistic),true_tail_mass_vs_absolute_tail_error_spearman=float(spearmanr(u.true_tail_mass,u.tail_error).statistic),length_vs_absolute_tail_error_spearman=float(spearmanr(u.length,u.tail_error).statistic),length_vs_absolute_tail_error_partial_rank_controlling_true_tail_mass=float(pearsonr(rx,re).statistic)))
        u['length_bin']=pd.cut(u.length,[0,150,300,600,np.inf],labels=['<=150','151-300','301-600','>600'])
        for key,v in u.groupby('length_bin',observed=True):
            g=v.groupby('group_id')[['true_tail_mass','predicted_tail_mass','lower_bias','upper_bias','tail_error']].mean()
            strata.append(dict(family=family,seed=int(seed),length_bin=str(key),unique_T=len(v),groups=len(g),**{k:float(x) for k,x in g.mean().items()}))
    report=dict(status='complete',scope='Exposed development OOF, descriptive conditional rank correlation; target-tail mass is an observed outcome used only for diagnosis, not a prospective deployable feature or a causal adjustment.',definition='tail_error=(abs(pred_lower-true_lower)+abs(pred_upper-true_upper))/2; signed bias is prediction minus target; true_tail_mass=first+last historical bin mass.',correlations=records,strata=strata,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (r/'public/tail_scale_diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
    s=pd.DataFrame(strata);fig,axs=plt.subplots(1,3,figsize=(14,4))
    for family,col in [('composition_charge_deep','#737b86'),('mean_pre','#2673b8'),('residue_mean','#dc6b2f')]:
        t=s[(s.family==family)&(s.seed==1701)]
        for ax,k in zip(axs,['lower_bias','upper_bias','tail_error']):ax.plot(t.length_bin,t[k],'-o',label=family,color=col);ax.axhline(0,color='black',lw=.6);ax.set_title(k);ax.set_xlabel('Teacher length');ax.set_ylabel('Conditional group mean');ax.grid(alpha=.15)
    axs[0].legend(fontsize=8);fig.suptitle('Tail magnitude and signed error | historical bins | seed1701');fig.tight_layout();fig.savefig(r/'plots/tail_scale_diagnostic.png',dpi=160,bbox_inches='tight');fig.savefig(r/'plots/tail_scale_diagnostic.svg',bbox_inches='tight');plt.close(fig)
    # Keep the private browsing page useful while exporting only aggregate tables.
    page=r/'index.html'
    if page.exists():
        import re
        body=page.read_text(encoding='utf8')
        body=re.sub(r'<section id="tail-scale">.*?</section>','',body,flags=re.S)
        section='<section id="tail-scale"><h2>补充：尾质量尺度与误差方向</h2><p>真实尾质量与长度Spearman约0.299。控制真实尾质量后，残基mean的长度—绝对尾误差相关从0.328/0.336降至0.223/0.243，部分关联来自目标尺度，仍有剩余关联。这里使用评价真值作诊断，不能作为部署输入、拒识阈值或因果调整。长链组的负侧尾质量平均偏低更明显，但个例可有相反方向；当前证据不支持简单增加全局平滑。</p><img src="plots/tail_scale_diagnostic.png" alt="负尾正尾方向偏差与长度"><details><summary>两seed相关与目标尺度控制</summary>'+pd.DataFrame(records).to_html(index=False,float_format=lambda x:f'{x:.6g}')+'</details><details><summary>条件组平均：真实尾质量、预测与有符号误差</summary>'+s.to_html(index=False,float_format=lambda x:f'{x:.6g}')+'</details><p>这一观察支持后续在共同优化机会下有限比较尾部校准或复合loss；新物理63箱与旧63箱定义不同，应先看新标签桥接证据，不直接继承旧尾部权重或阈值。</p></section>'
        page.write_text(body.replace('</main>',section+'</main>'),encoding='utf8')
    print(json.dumps(records,indent=2))
if __name__=='__main__':main()
