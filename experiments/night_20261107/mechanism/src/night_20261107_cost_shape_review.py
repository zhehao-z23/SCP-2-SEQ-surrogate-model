"""Independent aggregate-only decision recomputation, no fitting or sample reads."""
import hashlib,json
from pathlib import Path
R=Path(__file__).parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(name):return json.loads((R/name).read_bytes())
lp=load('night_20261107_shape_plan_v1.json');linear=load('night_20261107_shape_probe_summary.json');np=load('night_20261107_nonlinear_plan.json');nonlinear=load('night_20261107_nonlinear_summary.json')
files=['night_20261107_shape_plan_v1.json','night_20261107_shape_probes.py','night_20261107_shape_probe_summary.json','night_20261107_nonlinear_plan.json','night_20261107_nonlinear_probe.py','night_20261107_nonlinear_summary.json']
assert linear['plan_sha256']==sha(R/files[0]) and linear['code_sha256']==sha(R/files[1])
assert nonlinear['plan_sha256']==sha(R/files[3]) and nonlinear['code_sha256']==sha(R/files[4])
assert nonlinear['linear_trigger_sha256']==sha(R/files[2])==np['trigger_summary_sha256']
base={(x['method'],x['partition']):x for x in linear['results']};review=[]
for family,summary,key in [('linear',linear,'method'),('nonlinear',nonlinear,'arm')]:
 by={(x[key],x['partition']):x for x in summary['results']}
 for decision in summary['decisions']:
  name=decision[key];checks=[]
  for part in ['stop','evaluation']:
   b=base[('PMF_uniform_Bowley',part)];c=by[(name,part)]
   btp=sum(b['ranking'][side]['true_positive_T'] for side in ['negative','positive'])
   ctp=None if c['ranking'] is None else sum(c['ranking'][side]['true_positive_T'] for side in ['negative','positive'])
   ar=c['strong']['union']['Bowley_AE']['group_macro']/b['strong']['union']['Bowley_AE']['group_macro'];mr=c['overall']['Bowley_MSE']['group_macro']/b['overall']['Bowley_MSE']['group_macro']
   v={'partition':part,'rare_AE_reduction10pct':ar<=.9,'overall_MSE_within5pct':mr<=1.05,'ranking_not_lower':ctp is not None and ctp>=btp,'TP_delta':None if ctp is None else ctp-btp}
   original=next(x for x in decision['checks'] if x['partition']==part)
   assert v==original
   checks.append({**v,'rare_AE_ratio_to_PMF':ar,'overall_MSE_ratio_to_PMF':mr,'baseline_TP':btp,'probe_TP':ctp})
  eligible=(name.endswith('__raw') or name=='affine_scalar_raw') if family=='linear' else name=='raw'
  passed=eligible and all(x['rare_AE_reduction10pct'] and x['overall_MSE_within5pct'] and x['ranking_not_lower'] for x in checks) and any(x['TP_delta'] is not None and x['TP_delta']>=2 for x in checks)
  assert passed==decision['promising_for_strong_control_comparison']
  review.append({'family':family,'candidate':name,'eligible':eligible,'checks':checks,'promotion':passed})
assert not any(x['promotion'] for x in review)
assert linear['PMF_unchanged'] and nonlinear['PMF_unchanged'] and nonlinear['paired_initialization_batch_and_dropout_RNG_checks']==200
assert nonlinear['fits'][0]['initial_state_sha256']==nonlinear['fits'][1]['initial_state_sha256']
assert nonlinear['fits'][0]['initial_RNG_sha256']==nonlinear['fits'][1]['initial_RNG_sha256']
assert all(x['epochs']==200 and x['selected_epoch']==3 and x['checkpoint_prediction_replay_maxdiff']==0 for x in nonlinear['fits'])
out={'schema':'independent_shape_probe_static_and_decision_review_v1','status':'pass_with_scope_limits','source_refs':[{'file':f,'sha256':sha(R/f)} for f in files],'review_code_sha256':sha(Path(__file__)),'execution':'Local aggregate JSON arithmetic only. No new fit, protected-object access, sample table or checkpoint read, server mutation, or publication. This review is post-execution auditing, not preregistration.',
 'static_checks':{'linear_weighted_ridge':'C=X^T W X/sumW; D=X^T W centeredY/sumW; eigen solve (C+lambda I)^-1 D matches declared weighted-MSE ridge; train-only weighted centering and standardization, unpenalized intercept.','linear_lambda_selection':'Only raw-task stop group-MSE chooses lambda; exact ties prefer larger lambda. Both raw and binned target columns share the chosen lambda/X/scaler/weights. Evaluation is not accessed for lambda choice. This explicit asymmetry favors the intended raw task.','nonlinear_controls':'Shared hidden256, fit-only scaler and raw-target mean/std; identical initial state, optimizer initialization, shuffled batches and dropout RNG. Two target columns differ; no backbone or PMF update.','nonlinear_epoch_selection':'Epoch0 eligible; raw-task stop group-MSE only, improvement >1e-7; all200 epochs executed. Evaluation predictions are computed after fitting and checkpoint selection. Binned arm also uses raw-task stop selection, as declared.','eligibility_and_ranking':'Inherited finite predicted-PMF quartiles/IQR>=1; all current rows eligible. Clipped scalar used for value/threshold metrics, preclip scalar for fixed-budget ranking; constant control gets no informative retrieval credit. Raw eligibility is used only for truth.','no_direct_promotion':'Same engineering gate on stop and evaluation; no candidate replaces a whole-distribution model. Only raw-target arms (and declared affine control) are candidate-eligible.'},
 'recomputed_decisions':review,'nonlinear_selected_epochs':[x['selected_epoch'] for x in nonlinear['fits']],'nonlinear_paired_RNG_batch_checks_reported':200,'nonlinear_checkpoint_replay_maxdiff_reported':0,'unchanged_PMF_reported':True,
 'audit_boundary':'Source inspected and summary claims/hash binding checked. Independent review did not reload raw feature arrays, replay checkpoints, or independently recheck200 individual epoch hashes. Those are producer assertions bound by the inspected code and summary.',
 'interpretation':['No tested probe passes the frozen ranking-and-error gate. Nonlinear raw rare-shape MAE improves while fixed-budget summed TP changes -2/-1 on stop/evaluation; better regression is insufficient for this retrieval criterion.','Linear or one small nonlinear failure is not evidence that representation contains no information; finite support, accessibility, function class and prior representation selection remain alternatives.','Epoch3 was the stop-selected checkpoint out of200; this does not prove convergence or optimization exhaustion.','The backbone was previously stop-selected and this is exposed development data; no independent final-test or new-C/AFDB claim.','A scalar auxiliary prediction is not a self-consistent improved probability distribution or physical causal explanation.','Promotion thresholds are engineering decision rules, not significance tests.']}
dest=R/'night_20261107_cost_shape_independent_review.json'
with dest.open('x') as f:json.dump(out,f,indent=2,allow_nan=False)
print(json.dumps({'review_sha256':sha(dest),'decisions_checked':len(review),'all_promotion_false':True,'source_refs':out['source_refs']}))
