"""Read-only bounded metadata inventory; no labels, model arrays or production writes."""
import argparse, ast, hashlib, json, os
from pathlib import Path

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p): return json.loads(Path(p).read_text())
def ref(p): return {'path':str(p),'sha256':sha(p),'bytes':Path(p).stat().st_size}
def physical(c):
    p=Path('/sys/devices/system/cpu')/f'cpu{c}'/'topology'
    return (int((p/'physical_package_id').read_text()),int((p/'core_id').read_text()))
def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',type=Path,required=True);p.add_argument('--project',type=Path,required=True);p.add_argument('--physical',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    up=load(a.phase/'runtime_upgrade.json');release=Path(up['new_release']['path']).parent
    assert sha(release/'MANIFEST.json')==up['new_release']['sha256']
    files={r['relative']:r for r in load(release/'MANIFEST.json')['files']}
    controller=release/'tools/production_controller.py';assert sha(controller)==files['tools/production_controller.py']['sha256']
    text=controller.read_text();fn=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='cpu_pairs')
    fntext=ast.get_source_segment(text,fn)
    plan=load(a.phase/'phase.json')['plan'];cpus=sorted(os.sched_getaffinity(0));unique=[];seen=set()
    for c in cpus:
        key=physical(c)
        if key not in seen:unique.append(c);seen.add(key)
    slots=min(plan['cpu_slots'],64);representatives=unique[:slots]
    assert "capacity=dict(cpu_slots=min(phase['plan']['cpu_slots'],64),stage_concurrency=64,case_concurrency=64)" in text
    assert "cpus=cpus[:capacity['cpu_slots']]" in text and not plan.get('host_resource_policy')
    candidates=[172,173,174,175,176,177,178,179];candidate_physical={physical(c) for c in candidates}
    assert not candidate_physical & {physical(c) for c in representatives}
    processes=[]
    for d in Path('/proc').iterdir():
        if not d.name.isdigit():continue
        try:
            raw=(d/'cmdline').read_bytes();cg=(d/'cgroup').read_text();cmd=raw.replace(b'\0',b' ').decode(errors='replace')
            worker='/atlas_full_dC_' in cg and '/case_' in cg
            ctl='databank_production_20261102/controller/releases/' in cmd and 'production_controller.py' in cmd
            backup='databank_production_20261102' in cmd and 'progress' in cmd and 'backup' in cmd
            if not (worker or ctl or backup):continue
            affinity=sorted(os.sched_getaffinity(int(d.name)));overlap=sorted(candidate_physical & {physical(c) for c in affinity})
            if worker:assert not overlap
            processes.append({'pid':int(d.name),'role':'worker' if worker else 'controller' if ctl else 'backup','affinity':affinity,'overlap':overlap,'cmd_sha256':hashlib.sha256(raw).hexdigest()})
        except (OSError,ProcessLookupError):continue
    assert any(r['role']=='controller' for r in processes)
    accepted=list((a.physical/'accepted_development_labels').glob('*/qualification_receipt.json'))
    accept=[dict(ref(x),status=load(x)['status'],role=load(x)['role'],sequence_sha256=load(x)['source_identity']['sequence_sha256']) for x in accepted]
    prepared=[]
    for x in accepted:
        receipt=load(x);q=Path(receipt['sequence_and_structure']['path']);assert sha(q)==receipt['sequence_and_structure']['sha256'];d=load(q)
        prepared.append({'ref':ref(q),'keys':list(d),'nested_keys':{k:list(v) for k,v in d.items() if isinstance(v,dict)},'representation_manifest':load(receipt['label_representation']['path'])})
    roots=[a.project/'overnight_20261106/afdb',a.project/'pooling_20261105',a.project/'bridge_20261106/optimization']
    directories={str(r):[str(x.relative_to(r)) for x in r.glob('*')] for r in roots}
    cache_meta=[]
    for r in roots[:2]:
        for f in [r/'cache/receipt.json',r/'cache/identity.json',r/'records/cache_launch_full.json',r/'records/cache_launch_pilot.json']:
            if f.is_file():cache_meta.append({'ref':ref(f),'data':load(f)})
    result={'status':'passed','runtime_upgrade':dict(ref(a.phase/'runtime_upgrade.json'),value=up),'release_manifest':ref(release/'MANIFEST.json'),'controller':ref(controller),'cpu_pairs_function':fntext,'phase':ref(a.phase/'phase.json'),'plan_cpu_slots':slots,'production_representatives':representatives,'candidate_topology':{c:physical(c) for c in candidates},'live_processes':processes,'no_OS_exclusivity':True,'mem_available_line':next(s for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')),'accepted_development':accept,'prepared_metadata':prepared,'bounded_directories':directories,'cache_metadata':cache_meta,'runtime_environment_files':[str(x) for x in (a.project/'runtime').glob('*python*')]}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({'status':'passed','output':str(a.output),'sha256':sha(a.output),'accepted_labels':len(accept),'worker_processes':sum(r['role']=='worker' for r in processes),'runtime_sha256':sha(a.phase/'runtime_upgrade.json')}))
if __name__=='__main__':main()
