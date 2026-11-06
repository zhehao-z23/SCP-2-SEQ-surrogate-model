"""Read-only production cost census and bounded field/length metadata sample."""
import collections,datetime,hashlib,json,os,resource,sqlite3,time
from pathlib import Path
R=Path('/path/to/seq2scp/night_20261107/cost');P=Path('/path/to/new_atlas_production');F=P/'controller/runs/full_dC_09432fe5291464_pf7e254b465ebb7b9'
assert set(os.sched_getaffinity(0))=={174,175}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2)
def stats(x):
 x=sorted(float(v) for v in x if v is not None)
 if not x:return {'n':0}
 def q(a):
  i=(len(x)-1)*a;lo=int(i);return x[lo]+(x[min(lo+1,len(x)-1)]-x[lo])*(i-lo)
 return {'n':len(x),'sum':sum(x),'mean':sum(x)/len(x),'min':x[0],'median':q(.5),'p90':q(.9),'p95':q(.95),'max':x[-1]}
start=time.time();db=sqlite3.connect('file:'+str(F/'queue.sqlite')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row;db.execute('BEGIN')
cases=[dict(x) for x in db.execute('select sample_id,state,entry,science_receipt,updated_ns from cases')]
counts=dict(db.execute('select state,count(*) from cases group by state'));event_counts=dict(db.execute('select operation,count(*) from events group by operation'))
events={}
for z in db.execute("select sample_id,time_ns,operation,details from events where sample_id is not null order by seq"):
 k=z['sample_id'];v=events.setdefault(k,{'first':z['time_ns']/1e9,'last':z['time_ns']/1e9,'running':[],'science_complete':[],'complete':[],'recoveries':0})
 v['last']=z['time_ns']/1e9
 if z['operation']=='dead_owner_recovered':v['recoveries']+=1
 if z['operation']=='transition':
  to=json.loads(z['details']).get('to')
  if to in ['running','science_complete','complete']:v[to].append(z['time_ns']/1e9)
db.rollback();db.close()
complete=[x for x in cases if x['state']=='complete'];sample=sorted(complete,key=lambda x:hashlib.sha256(x['sample_id'].encode()).hexdigest())[:128]
latest=sorted(complete,key=lambda x:x['updated_ns'])[-64:];selected={x['sample_id'] for x in sample+latest}
private=[];issues=[];allreceipts=[];fieldreceipts=[];status=collections.Counter();stage=collections.defaultdict(list);implementations=collections.Counter()
for i,c in enumerate(cases):
 if c['state'] not in ['complete','failed']:continue
 sid=c['sample_id'];entry=json.loads(c['entry']);er=Path(entry['output'])/'execution_receipts';receipts=[]
 for p in sorted(er.glob('*.json')):
  try:
   z=json.loads(p.read_text());assert z.get('sample_id')==sid
   if 'wall_seconds' not in z or 'cpu' not in z:issues.append({'sample_id':sid,'file':str(p),'reason':'missing_cost_fields'});continue
   row={'path':str(p),'sha256':sha(p),'sample_id':sid,'status':z['status'],'mode':z['mode'],'wall_seconds':z['wall_seconds'],'cpu_seconds':z['cpu']['total_seconds'],'started_at_utc':z['started_at_utc'],'finished_at_utc':z.get('finished_at_utc'),'admission_wait_seconds':z.get('admission_timing',{}).get('admission_wait_seconds'),'service_elapsed_seconds':z.get('admission_timing',{}).get('service_elapsed_seconds'),'wait_observation_complete':z.get('admission_timing',{}).get('wait_observation_complete')}
   status[(row['mode'],row['status'])]+=1;implementations[z['timing_implementation_sha256']]+=1;receipts.append(row);allreceipts.append(row)
   for s in z.get('stages',[]):stage[s['name']].append({'wall':s['wall_seconds'],'cpu':s['cpu']['total_seconds'],'wait':s.get('admission_timing',{}).get('admission_wait_seconds',0),'cached':s.get('completion_existed_before')})
  except Exception as exc:issues.append({'sample_id':sid,'path':str(p),'error':repr(exc)})
 e=events.get(sid,{})
 item={'sample_id':sid,'state':c['state'],'receipts':len(receipts),'captured_cpu_seconds':sum(x['cpu_seconds'] for x in receipts),'captured_wall_task_seconds':sum(x['wall_seconds'] for x in receipts),'captured_service_task_seconds':sum(x['service_elapsed_seconds'] or 0 for x in receipts),'captured_wait_task_seconds':sum(x['admission_wait_seconds'] or 0 for x in receipts),'recoveries':e.get('recoveries',0),'queue_first_to_complete_seconds':max(e['complete'])-e['first'] if e.get('complete') else None,'first_run_to_science_complete_seconds':max(e['science_complete'])-min(e['running']) if e.get('science_complete') and e.get('running') else None,'latest64':sid in {v['sample_id'] for v in latest},'uniform_SHA128':sid in {v['sample_id'] for v in sample}}
 if sid in selected:
  reqp=Path(entry['request']['path']);req=json.loads(reqp.read_text());assert sha(reqp)==entry['request']['sha256'];item['C_length']=len(req['model_input']['sequence']);item['request_sha256']=entry['request']['sha256']
  science=json.loads(c['science_receipt']);assert science['document']['accepted'];fields=science['document']['result']['all_field_receipts'];item['PB_fields']=len(fields);item['APBS_wall_task_seconds']=0;item['APBS_peak_RSS_bytes']=0
  for ref in fields:
   fp=Path(ref['path']);assert sha(fp)==ref['sha256'];v=json.loads(fp.read_text());assert v['status']=='completed';cost=v['result'];item['APBS_wall_task_seconds']+=cost['wall_seconds'];item['APBS_peak_RSS_bytes']=max(item['APBS_peak_RSS_bytes'],cost['peak_RSS_bytes']);fieldreceipts.append({'path':str(fp),'sha256':ref['sha256'],'attempt':v['PB_attempt_id'],'wall_seconds':cost['wall_seconds'],'threads':cost['threads'],'peak_RSS_bytes':cost['peak_RSS_bytes']})
 private.append(item)
 if i%500==0:print(json.dumps({'metadata_cases_processed':i}),flush=True)
ld=sqlite3.connect('file:'+str(P/'project/records/budget/PB_attempts.sqlite')+'?mode=ro',uri=True)
pb=[{'phase':a,'status':b,'n':c} for a,b,c in ld.execute('select phase,status,count(*) from attempts group by phase,status')]
def cohort(rows):return {'cases':len(rows),'with_execution_receipts':sum(x['receipts']>0 for x in rows),'CPU_seconds_per_case':stats(x['captured_cpu_seconds'] for x in rows if x['receipts']),'captured_pipeline_task_wall_seconds_per_case':stats(x['captured_wall_task_seconds'] for x in rows if x['receipts']),'service_task_seconds_per_case':stats(x['captured_service_task_seconds'] for x in rows if x['receipts']),'stage_pool_wait_task_seconds_per_case':stats(x['captured_wait_task_seconds'] for x in rows if x['receipts']),'queue_first_to_complete_seconds':stats(x['queue_first_to_complete_seconds'] for x in rows),'first_run_to_science_complete_seconds':stats(x['first_run_to_science_complete_seconds'] for x in rows),'recovered_case_count':sum(x['recoveries']>0 for x in rows),'C_length':stats(x.get('C_length') for x in rows),'APBS_wall_task_seconds':stats(x.get('APBS_wall_task_seconds') for x in rows),'APBS_peak_RSS_bytes':stats(x.get('APBS_peak_RSS_bytes') for x in rows),'PB_fields':stats(x.get('PB_fields') for x in rows)}
out={'schema':'normal_production_cost_census_v1','status':'complete','started_unix':start,'finished_unix':time.time(),'queue_snapshot_counts':counts,'queue_event_counts':event_counts,'PB_ledger_counts_at_end':pb,'completed_census':cohort([x for x in private if x['state']=='complete']),'uniform_SHA128_completed':cohort([x for x in private if x['uniform_SHA128']]),'latest64_completed':cohort([x for x in private if x['latest64']]),'failed_state_captured_costs':cohort([x for x in private if x['state']=='failed']),'execution_receipt_status_counts':[{'mode':a,'status':b,'n':v} for (a,b),v in status.items()],'timing_implementation_counts':dict(implementations),'stage_costs':{k:{'wall_task_seconds':stats(x['wall'] for x in v),'CPU_seconds':stats(x['cpu'] for x in v),'admission_wait_seconds':stats(x['wait'] for x in v),'completion_existed_before_count':sum(bool(x['cached']) for x in v)} for k,v in stage.items()},'issues_count':len(issues),'cost_definitions':{'CPU_seconds':'Captured self user+system plus reaped children user+system in timed production execution receipts. Not inferred from APBS wall; stages are not added again to total.','pipeline_wall':'Sum of captured invocation elapsed times, not elapsed time of parallel full cohort. Queue intervals also include waiting/recovery/publication.','scope':'Completed snapshot biased toward finished/supported cases. No cost guarantee for remaining or much larger proteins. No archive object/label/field arrays read. Missing or killed unreceipted work and auxiliary controller/publication processes are not reconstructed.','failed':'Failed queue states and recovery events are not automatically failed PB attempts; ledger status counts are separate.'},'source_code_sha256':sha(__file__),'metadata_review_CPU_seconds':resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime}
put(R/'private/production_cases.json',private);put(R/'private/production_execution_refs.json',allreceipts);put(R/'private/production_field_refs.json',fieldreceipts);put(R/'private/production_issues.json',issues);put(R/'results/production_cost_public.json',out)
print(json.dumps({'status':'complete','complete_cases':len(complete),'receipts':len(allreceipts),'field_refs':len(fieldreceipts),'issues':len(issues),'census_cpu':out['completed_census']['CPU_seconds_per_case'],'census_wall':out['completed_census']['captured_pipeline_task_wall_seconds_per_case']}),flush=True)
