"""Build anonymous evidence projections and scientific figures from completed probes."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(p):
    return json.loads(Path(p).read_bytes())


def write(p, data):
    Path(p).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--plots', action='store_true')
    a = ap.parse_args()
    out = a.root/'night_20261107_mechanism'
    out.mkdir(exist_ok=True)
    sources = {
        'linear': ('night_20261107_shape_probe_summary.json', 'e7c0f3d6c66d4f9da3947a3c9f6c1ade4ce4ce6727a2edbe7c84c3ce93de1b41'),
        'nonlinear': ('night_20261107_nonlinear_summary.json', 'c56fe6c6bb5e3c6d2bf813b7d90ec399213281297270a3a1a42c075618199fd9'),
        'extract': ('night_20261107_shape_extract_summary.json', '5de0fefc16638e86dc8a8292ec32228b313c154ff31c2094583b98b287b8245c')}
    src={}
    for name,(file,digest) in sources.items():
        assert sha(a.root/file)==digest
        src[name]=load(a.root/file)
    rows=[]
    for family in ['linear','nonlinear']:
        for r in src[family]['results']:
            item={'method':r['method'] if family=='linear' else 'hidden256_MLP__'+r['arm'],
                  'partition':r['partition'], 'overall':r['overall'], 'strong':r['strong'],
                  'threshold':{s:{'TP':r[s]['row_events']['TP'], 'FP':r[s]['row_events']['FP'],
                                  'FN':r[s]['row_events']['FN']} for s in ['negative','positive']},
                  'ranking':r['ranking']}
            rows.append(item)
    curves={}
    for arm in ['raw','binned']:
        path=a.root/(arm+'_curve.json')
        fit=next(f for f in src['nonlinear']['fits'] if f['arm']==arm)
        assert sha(path)==fit['curve_sha256']
        curves[arm]=[{k:r[k] for k in ['epoch','lr','online_normalized_target_MSE','stop_raw_group_MSE','elapsed_seconds']} for r in load(path)]
    write(out/'curves.json',curves)
    result={
        'schema':'shape_information_accessibility_anonymous.v1', 'status':'complete',
        'source_summary_sha256':{k:v[1] for k,v in sources.items()},
        'question':'Does correcting score scale, changing scalar supervision, or increasing frozen-readout capacity recover strong-shape screening information?',
        'development_design':{'fit_folds':[2,3,4], 'stop_fold':1, 'evaluation_fold':0, 'seed':1701,
            'head_seed':1707,'final_independent_test':False,'backbone_selection':'Historical stop group-H checkpoint 104; reused, frozen.',
            'probe_selection':'Ridge lambda and neural epoch selected only on stop raw group-MSE; development evaluation is exposed.',
            'truth':'Raw QC Bowley <=-0.30 or >=+0.30 with raw IQR>=1; not the complete historical percentile screen.',
            'ranking':'Separate negative and positive top ceil(1% unique T); canonical fixed view score; any-view truth.',
            'support':{'train':{'T':13162,'groups':4800,'strong_T':49,'strong_groups':48},'stop':{'T':4389,'groups':1600,'strong_T':13,'strong_groups':12},'evaluation':{'T':4364,'groups':1599,'strong_T':16,'strong_groups':16}}},
        'controls':src['linear']['controls'],
        'ridge_selection':src['linear']['ridge_selection'],
        'neural_fits':src['nonlinear']['fits'],
        'pairing':{'neural_init_and_rng_equal':True,'batch_and_dropout_checks':200,'checkpoint_prediction_replay_maxdiff':0.0,'PMF_unchanged':True},
        'results':rows,
        'promotion_gate':'Rare group-MAE at least 10% lower in stop and evaluation; whole-cohort group-MSE no more than 5% worse in both; sum of directional budget TP not lower in either and at least 2 higher in one. Raw-target candidates only.',
        'decisions':{'linear':src['linear']['decisions'],'nonlinear':src['nonlinear']['decisions'],'promoted':[], 'next':'Stop this bounded frozen-readout branch; retain current distribution model and strong charge control. No head/threshold rescue.'},
        'cost':{'feature_extraction_seconds':26.8851,'ridge_process_seconds':src['linear']['process_seconds'],'nonlinear_process_seconds':src['nonlinear']['process_seconds'],'new_backbone_fits':0,'ridge_vectors':24,'nonlinear_heads':2,'nonlinear_parameters_each':33025,'nonlinear_epochs_each':200,'neural_peak_GPU_bytes':src['nonlinear']['peak_GPU_bytes'], 'meaning':'Observed process durations; not CPU or GPU busy-time accounting.'},
        'limits':['Single exposed development fold/backbone; stop selected the backbone before the heads.','Rare support is small; no statistical significance or universal architecture claim.','Scalar probes leave PMF unchanged and cannot establish restored distribution shape.','Failed linear or small nonlinear probes do not prove the representation lacks relevant information.','Raw-target selection favors the intended raw task; binned partner shares hyperparameters and normalization.'],
        'literature':[{'url':'https://arxiv.org/pdf/1610.01644','title':'Understanding intermediate layers using linear classifier probes','read_scope':'Introduction and Section 3','use':'Independent frozen probes diagnose accessibility to the specified probe; not total information or causal mechanism.'},{'url':'https://arxiv.org/pdf/1206.4661','title':'Predicting accurate probabilities with a ranking loss','read_scope':'Sections 1–3','use':'Separate score-value and ranking questions. Its binary calibration theory is not a guarantee for continuous Bowley regression.'}]}
    # Public input is a path-only projection; scientific output retains original execution provenance.
    result['source_summary_sha256']['extract'] = 'b4cdf785c67d8f47a52eb8517e8a8f77fce0183b77c57f99beb1a8875bcf4a30'
    write(out/'summary.json',result)
    if a.plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
        methods=['PMF_uniform_Bowley','affine_scalar_raw','hidden256__raw','hidden256_MLP__raw','hidden256_MLP__binned']
        labels=['PMF-derived','Affine raw','Linear hidden raw','MLP hidden raw','MLP hidden binned']
        colors=['#34495e','#2a9d8f','#4679bd','#e48a35','#929aa5']
        table={(r['method'],r['partition']):r for r in rows}
        fig,axs=plt.subplots(2,2,figsize=(12.8,7.6),layout='constrained')
        for j,part in enumerate(['stop','evaluation']):
            rr=[table[(m,part)] for m in methods]
            vals=[r['strong']['union']['Bowley_AE']['group_macro'] for r in rr]
            tps=[sum(r['ranking'][s]['true_positive_T'] for s in ['negative','positive']) for r in rr]
            axs[0,j].barh(labels,vals,color=colors)
            axs[0,j].invert_yaxis();axs[0,j].set_xlim(0,.225)
            axs[0,j].set_xticks([0,.05,.10,.15,.20])
            for i,v in enumerate(vals):axs[0,j].text(v+.002,i,f'{v:.3f}',va='center')
            axs[0,j].set_xlabel('Strong-shape group MAE (lower is better)')
            axs[0,j].set_title(('Stop: 13 strong T / 12 groups' if part=='stop' else 'Development evaluation: 16 strong T / 16 groups'))
            axs[1,j].barh(labels,tps,color=colors)
            axs[1,j].invert_yaxis();axs[1,j].set_xlim(0,13)
            for i,(v,r) in enumerate(zip(tps,rr)):
                n,p=(r['ranking'][s]['true_positive_T'] for s in ['negative','positive'])
                axs[1,j].text(v+.15,i,f'{v} = {n} neg + {p} pos',va='center',fontsize=10)
            axs[1,j].set_xlabel('TP across two separate 1% budgets (44 T per direction)')
        fig.suptitle('Smaller shape error did not improve extreme-object retrieval',fontsize=18)
        fig.savefig(out/'task_tradeoff.png',dpi=180);fig.savefig(out/'task_tradeoff.svg');plt.close(fig)
        fig,axs=plt.subplots(1,2,figsize=(12.8,4.8),layout='constrained')
        for arm,color in [('raw','#e48a35'),('binned','#4679bd')]:
            cr=curves[arm];ep=[r['epoch'] for r in cr]
            axs[0].plot(ep,[r['online_normalized_target_MSE'] for r in cr],color=color,label=arm+' supervision')
            axs[1].plot(ep,[r['stop_raw_group_MSE'] for r in cr],color=color,label=arm+' supervision')
            fit=next(f for f in result['neural_fits'] if f['arm']==arm)
            axs[1].scatter([fit['selected_epoch']],[fit['stop_raw_MSE']],color=color,s=55,zorder=5)
        for ax in axs:
            ax.set_xlabel('Epoch (all 200 observed)');ax.legend(frameon=False)
            ax.axvline(80,color='#cccccc',ls=':',zorder=0);ax.axvline(140,color='#cccccc',ls=':',zorder=0)
        axs[0].set_title('Online training objective');axs[0].set_ylabel('Weighted normalized target MSE')
        axs[1].set_title('Common validation metric; both select epoch 3');axs[1].set_ylabel('Stop raw Bowley group-MSE')
        fig.suptitle('Continuing optimization did not improve selected validation performance',fontsize=16)
        fig.savefig(out/'training_curves.png',dpi=180);fig.savefig(out/'training_curves.svg');plt.close(fig)
    write(out/'export_receipt.json',{'source':str(Path(__file__).name),'code_sha256':sha(__file__),'outputs':{p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='export_receipt.json'},'privacy':'Only aggregate metrics and curves; no sample identities, sequences, targets or weights.'})
    print(json.dumps({'summary_sha256':sha(out/'summary.json'),'output':str(out),'plots':a.plots}))


if __name__=='__main__':main()
