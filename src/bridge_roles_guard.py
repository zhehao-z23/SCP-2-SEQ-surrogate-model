"""Fail-closed preflight for future data/model jobs; invoke before loading labels/models.

Requests are JSON: {"operation":"train", "stage":"development",
"sequence_sha256":[...], "aliases":[...]}. Sequence hashes are mandatory.
Use the pinned contract SHA from the registry receipt, never compute it dynamically
as a substitute for a previously registered pin. This is a policy preflight, not an
OS access-control system; launchers must require its successful receipt.
"""
import argparse, hashlib, json, sys, time
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()

def check(contract_path, pinned_sha, request):
    contract_path=Path(contract_path)
    if not contract_path.is_file() or sha(contract_path)!=pinned_sha:
        raise PermissionError('missing or unpinned contract')
    contract=json.loads(contract_path.read_text())
    if contract.get('schema')!='bridge_guard_contract.v1' or not contract.get('search_complete'):
        raise PermissionError('incomplete relationship registry')
    if not contract.get('locks'): raise PermissionError('missing source locks')
    for path,expected in contract['locks'].items():
        if not Path(path).is_file() or sha(path)!=expected:
            raise PermissionError('missing or changed source/group/role/search lock: '+path)
    ids=json.loads(Path(contract['identities_path']).read_text())
    if request.get('operation') not in {'train','stop','model_select','model_eval','read_target','predict'}:
        raise PermissionError('unknown operation')
    if request.get('stage') not in {'development','final_release'}:
        raise PermissionError('explicit stage required')
    if request['stage']=='final_release':
        # This provisional registry cannot grant final release use. A new verified
        # schema and pinned registry must be issued after full closure is available.
        raise PermissionError('final release group/profile closure not certified by this registry')
    sequences=request.get('sequence_sha256')
    aliases=request.get('aliases',[])
    if not isinstance(sequences,list) or not sequences or any(not isinstance(x,str) for x in sequences):
        raise PermissionError('nonempty sequence identity list required')
    if not isinstance(aliases,list) or any(not isinstance(x,str) for x in aliases):
        raise PermissionError('invalid alias identity list')
    if set(sequences)-set(ids['known_sequence_hashes']): raise PermissionError('unregistered sequence identity')
    if set(aliases)-set(ids['known_aliases']): raise PermissionError('unregistered alias identity')
    if set(sequences)&set(ids['protected_sequence_hashes']): raise PermissionError('blinded protected component')
    if set(aliases)&set(ids['protected_aliases']): raise PermissionError('blinded protected alias/component')
    return {'status':'passed_bridge_protection_preflight','development_only':True,'existing_release_role_gate_still_required':True,'operation':request['operation'],
            'sequence_count':len(sequences),'alias_count':len(aliases),'contract_sha256':pinned_sha,
            'request_sha256':hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest(),
            'final_closure_ready':False,'checked_unix':time.time()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--contract',required=True)
    p.add_argument('--contract-sha256',required=True);p.add_argument('--request',required=True)
    p.add_argument('--receipt',required=True);a=p.parse_args()
    destination=Path(a.receipt)
    if destination.exists(): raise RuntimeError('refuse receipt overwrite')
    try:
        receipt=check(a.contract,a.contract_sha256,json.loads(Path(a.request).read_text()))
    except Exception as exc:
        receipt={'status':'denied','reason':str(exc),'checked_unix':time.time()}
        destination.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));sys.exit(2)
    destination.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))

if __name__=='__main__':main()
