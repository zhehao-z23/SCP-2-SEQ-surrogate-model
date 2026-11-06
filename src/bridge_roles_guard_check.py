"""Exercise real private registry decisions plus isolated lock-corruption fixtures."""
import argparse, importlib.util, json, tempfile
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--guard',required=True);p.add_argument('--registry',required=True);p.add_argument('--output',required=True);a=p.parse_args()
spec=importlib.util.spec_from_file_location('guard',a.guard);g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
r=Path(a.registry);contract=r/'guard_contract.json';pin=g.sha(contract)
ids=json.loads((r/'guard_identities_private.json').read_text());cases=[]
allowed=next(h for h in ids['known_sequence_hashes'] if h not in set(ids['protected_sequence_hashes']))
base={'operation':'train','stage':'development','sequence_sha256':[allowed]}
def run(name, req, want, c=contract, sha=pin):
    try: g.check(c,sha,req); actual=True;reason=None
    except Exception as exc:actual=False;reason=str(exc)
    assert actual==want,(name,actual,reason)
    cases.append({'case':name,'expected_allowed':want,'passed':True,'reason':reason})
run('known_development',base,True)
run('unknown_identity',dict(base,sequence_sha256=['0'*64]),False)
run('final_closure_missing',dict(base,stage='final_release'),False)
run('unpinned_contract',base,False,sha='0'*64)
run('empty_identity',dict(base,sequence_sha256=[]),False)
if ids['protected_sequence_hashes']:
    for op in ['train','stop','model_select','model_eval','read_target','predict']:
        run('reserved_'+op,dict(base,operation=op,sequence_sha256=[ids['protected_sequence_hashes'][0]]),False)
if ids['protected_aliases']: run('reserved_alias',dict(base,aliases=[ids['protected_aliases'][0]]),False)
# Only temporary synthetic files are changed; the actual registry stays immutable.
with tempfile.TemporaryDirectory(prefix='bridge_guard_check_') as t:
    t=Path(t);lock=t/'groupmap';lock.write_text('original')
    c=json.loads(contract.read_text());c['locks']={str(lock):g.sha(lock)}
    cp=t/'contract.json';cp.write_text(json.dumps(c));sp=g.sha(cp)
    lock.write_text('changed');run('changed_groupmap_lock',base,False,c=cp,sha=sp)
    lock.unlink();run('missing_search_or_source_lock',base,False,c=cp,sha=sp)
    c['search_complete']=False;cp.write_text(json.dumps(c));run('incomplete_search',base,False,c=cp,sha=g.sha(cp))
    # An unchanged contract pin must detect altered protected identities too.
    fixture=t/'identities.json';fixture.write_text(json.dumps(ids))
    c=json.loads(contract.read_text());c['identities_path']=str(fixture);c['locks']={str(fixture):g.sha(fixture)}
    cp.write_text(json.dumps(c));sp=g.sha(cp)
    altered=dict(ids,protected_sequence_hashes=[],protected_aliases=[])
    fixture.write_text(json.dumps(altered))
    run('tampered_protected_identity_lists',base,False,c=cp,sha=sp)
out=Path(a.output);assert not out.exists();out.write_text(json.dumps({'status':'passed','contract_sha256':pin,'guard_sha256':g.sha(a.guard),'cases':cases},indent=2)+'\n')
print(json.dumps({'status':'passed','cases':len(cases),'protected_cases_tested':bool(ids['protected_sequence_hashes'])}))
