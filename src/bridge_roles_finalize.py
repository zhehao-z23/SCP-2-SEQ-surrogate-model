"""Private bridge role registry. Reads identities/relations only, never labels/predictions."""
import argparse, collections, hashlib, json, time
from pathlib import Path
import pandas as pd

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()

def write(p, obj):
    p = Path(p)
    assert not p.exists(), p
    p.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')

class DSU:
    def __init__(self): self.p = {}; self.n = {}
    def find(self, x):
        if x not in self.p: self.p[x] = x; self.n[x] = 1
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b: return
        if self.n[a] < self.n[b]: a, b = b, a
        self.p[b] = a; self.n[a] += self.n[b]

def main():
    a = argparse.ArgumentParser()
    a.add_argument('--project-root', type=Path, required=True)
    a.add_argument('--search', type=Path, required=True)
    a.add_argument('--output', type=Path, required=True)
    a.add_argument('--guard', type=Path, required=True)
    a.add_argument('--exposure-supplement', type=Path, required=True)
    args = a.parse_args(); root, search, out = args.project_root, args.search, args.output
    assert not out.exists(); out.mkdir()
    plan = json.loads((search/'plan.json').read_text())
    receipt = json.loads((search/'receipt.json').read_text())
    assert receipt['status'] == 'complete'
    assert receipt['plan_sha256'] == sha(search/'plan.json')
    assert {r['direction'] for r in receipt['directions']} == set(plan['directions'])
    paths = {k: root/v['relative'] for k,v in plan['sources'].items()}
    for k,p in paths.items(): assert sha(p) == plan['sources'][k]['sha256'], k
    for d in receipt['directions']: assert sha(search/(d['direction']+'.tsv')) == d['sha256']
    dsu = DSU(); refs = json.loads((search/'reference_identity_private.json').read_text())
    identity_nodes = collections.defaultdict(list); search_nodes = collections.defaultdict(list)
    old_nodes = set(); c_nodes = set()
    for x in refs:
        n = x['node']; dsu.find(n)
        identity_nodes[x['identity_sha256']].append(n); search_nodes[x['search_sha256']].append(n)
        (old_nodes if n.startswith('o_') else c_nodes).add(n)
    groups = pd.read_parquet(paths['old_groups'])
    assert len(groups) == 187939 and groups.sequence_sha256.is_unique
    for h,g in zip(groups.sequence_sha256, groups.group_id): dsu.union('o_'+h, 'g_'+str(g))
    panel = json.loads((search/'panel_manifest_private.json').read_text())['records']
    exposed_nodes = set()
    for x in panel:
        h=x.get('sequence_sha256')
        if not h: continue
        n='p_'+h; dsu.find(n); identity_nodes[h].append(n); search_nodes[h].append(n)
        if x['exposed']: exposed_nodes.add(n)
    for mapping in (identity_nodes, search_nodes):
        for ns in mapping.values():
            for n in ns[1:]: dsu.union(ns[0], n)
    # Exact authoritative alias matches are conservatively joined to old identities.
    # Missing/ambiguous identities are not interpreted as evidence of independence.
    manifest = pd.read_parquet(paths['old_manifest'], columns=['sample_id','parent_id','source_sequence_id','sequence_sha256','teacher_sequence_sha256'])
    old_alias = collections.defaultdict(set)
    for row in manifest.itertuples(index=False):
        nodes = ['o_'+h for h in (row.sequence_sha256,row.teacher_sequence_sha256) if isinstance(h,str) and 'o_'+h in old_nodes]
        for key in (row.sample_id,row.parent_id,row.source_sequence_id):
            if isinstance(key,str) and key: old_alias[key].update(nodes)
    aliases = {}; alias_joins = 0
    for line in paths['aliases'].open():
        n,key=line.rstrip('\n').split('\t'); assert n in c_nodes
        assert key not in aliases; aliases[key]=n
        for old in old_alias.get(key, ()):
            dsu.union(n,old); alias_joins += 1
    assert len(aliases) == 706552
    hit_counts = {}
    for stage in receipt['directions']:
        raw=accepted=0
        for line in (search/(stage['direction']+'.tsv')).open():
            q,t,ident,qcov,tcov,qlen,tlen,evalue=line.strip().split('\t')
            assert q in dsu.p and t in dsu.p
            raw += 1
            coverage=float(qcov) if int(qlen)<=int(tlen) else float(tcov)
            if float(ident)>=.3 and coverage>=.8 and float(evalue)<=.001:
                dsu.union(q,t); accepted += 1
        hit_counts[stage['direction']]={'raw':raw,'accepted':accepted}
    supplement=json.loads(args.exposure_supplement.read_text(encoding='utf-8-sig'))
    assert supplement['status']=='complete' and not supplement['unresolved_accessions']
    assert set(supplement['receipts'])=={'candidate_to_exposed','exposed_to_candidate'}
    assert all(x['returncode']==0 for x in supplement['receipts'].values())
    for x in supplement['exposure_records']:
        h=x['sequence_sha256'];n='e_'+h;dsu.find(n);exposed_nodes.add(n)
        for other in identity_nodes.get(h,[]):dsu.union(n,other)
        for other in search_nodes.get(h,[]):dsu.union(n,other)
        identity_nodes[h].append(n)
    for x in supplement['accepted_hits']:
        assert x['query'] in dsu.p and x['target'] in dsu.p
        dsu.union(x['query'],x['target'])
    old_components={dsu.find(n) for n in old_nodes}
    exposed_components={dsu.find(n) for n in exposed_nodes}
    eligible=[x for x in panel if x['source']=='screen_v2' and x['input_eligible']]
    reserved={dsu.find('p_'+x['sequence_sha256']) for x in eligible
              if dsu.find('p_'+x['sequence_sha256']) not in old_components|exposed_components}
    supplemented={x['sequence_sha256'] for x in supplement['queries']}
    # Every sequence in a provisional reserved panel component must have the
    # supplemental historical exposure search, not just one accession in a family.
    assert {n[2:] for n in dsu.p if n.startswith('p_') and dsu.find(n) in reserved}<=supplemented
    rows=[]
    for x in panel:
        h=x.get('sequence_sha256'); comp=dsu.find('p_'+h) if h else None
        old=comp in old_components; exposed=comp in exposed_components
        if comp in reserved: role='reserved_pending_final_closure'
        elif x['exposed']: role='development_or_challenge_previously_exposed'
        elif x['input_eligible']: role='development_related_to_old_or_exposed'
        else: role='input_ineligible_or_unresolved_no_training'
        rows.append(dict(x,component=comp,role=role,related_old_reference=old,related_exposed_panel=exposed,
                         target_or_prediction_read_by_this_lane=False))
    protected_nodes={n for n in dsu.p if dsu.find(n) in reserved}
    protected_hashes={h for h,ns in identity_nodes.items() if any(n in protected_nodes for n in ns)}
    protected_hashes.update(h for h,ns in search_nodes.items() if any(n in protected_nodes for n in ns))
    protected_aliases={k for k,n in aliases.items() if n in protected_nodes}
    protected_aliases.update(x['accession'] for x in rows if x['role']=='reserved_pending_final_closure')
    component_details=[]
    for comp in sorted(reserved):
        ns={n for n in protected_nodes if dsu.find(n)==comp}
        component_details.append({'component':comp,'panel_accessions':sorted({x['accession'] for x in rows if x['component']==comp}),
          'related_newC_sequence_count':sum(n.startswith('c_') for n in ns),
          'related_newC_alias_count':sum(n in ns for n in aliases.values()),
          'old_reference_count':sum(n.startswith('o_') for n in ns)})
    write(out/'roles_private.json',{'schema':'bridge_roles.v1','records':rows,'reserved_components':component_details,
       'additional_historical_exposure_records':[dict(x,role='development_or_challenge_previously_exposed') for x in supplement['exposure_records']]})
    known_aliases=set(aliases)|set(old_alias)|{x['accession'] for x in panel if x.get('accession')}|{x['accession'] for x in supplement['exposure_records']}
    guard={'schema':'bridge_guard_identities.v1','known_sequence_hashes':sorted(set(identity_nodes)|set(search_nodes)),
           'known_aliases':sorted(known_aliases),'protected_sequence_hashes':sorted(protected_hashes),
           'protected_aliases':sorted(protected_aliases),'protected_nodes':sorted(protected_nodes),
           'final_closure_ready':False}
    write(out/'guard_identities_private.json',guard)
    locks={str(p):sha(p) for p in paths.values()}
    for p in [search/'plan.json', search/'receipt.json',search/'panel_manifest_private.json',search/'reference_identity_private.json',out/'roles_private.json',out/'guard_identities_private.json',Path(__file__),args.guard,args.exposure_supplement,root/'bridge_20261106/physical/inputs/pilots.json']: locks[str(p)]=sha(p)
    for stage in receipt['directions']:
        p=search/(stage['direction']+'.tsv'); locks[str(p)]=sha(p)
    contract={'schema':'bridge_guard_contract.v1','created_unix':time.time(),'locks':locks,
      'identities_path':str(out/'guard_identities_private.json'),'roles_path':str(out/'roles_private.json'),
      'search_complete':True,'final_closure_ready':False,
      'required_before_final_release':['current complete new-release group map','current sequence and profile closure including protected panel','exposure audit and explicit blinded role approval'],
      'protection':'No target read, prediction, training, stopping, model selection or evaluation on reserved component while blinded.',
      'additional_required_gate':'This preflight certifies only bridge reservation exclusion. Existing release eligibility and train/stop/eval role rules remain mandatory; a pass does not grant general training eligibility.',
      'external_exposure_snapshot':{'sources':supplement['sources'],'requirement':'Any new historical exposure or change to these G3 inputs requires a new audit and registry. Snapshot hashes certify the audited state, not live remote-file monitoring.'},
      'scope':'Small AF panel bidirectional sequence relations against old views and newC; inherited old groups and all newC aliases. Full newC-to-old/profile closure NOT completed. Heuristic non-hit is not proof of distant-family or pLM-pretraining independence.'}
    write(out/'guard_contract.json',contract)
    summary={'status':'complete_provisional_roles','input_eligible_v2':len(eligible),'reserved_candidate_count':sum(x['source']=='screen_v2' and x['input_eligible'] and x['component'] in reserved for x in rows),
      'reserved_component_count':len(reserved),'protected_newC_sequences':sum(n.startswith('c_') for n in protected_nodes),
      'protected_newC_aliases':len({k for k,n in aliases.items() if n in protected_nodes}),
      'old_reference_nodes':len(old_nodes),'newC_reference_nodes':len(c_nodes),'newC_alias_count':len(aliases),
      'exact_old_alias_joins':alias_joins,'hits':hit_counts,'roles':dict(collections.Counter(x['role'] for x in rows)),
      'supplemental_exposure':{'resolved_accessions':supplement['resolved_accessions'],'unique_sequences':supplement['unique_exposed_sequences'],'unresolved_accessions':len(supplement['unresolved_accessions']),'accepted_hits':len(supplement['accepted_hits']),'candidate_queries':len(supplement['queries']),'historical_sequences_transferred':False},
      'final_closure_ready':False,'no_target_or_prediction_reads':True,'search_scope':contract['scope'],
      'selection_bias':'Input-screened sample from existing biased AF pool; not a representative AFDB population or pLM-unseen set.',
      'guard_contract_sha256':sha(out/'guard_contract.json'),'script_sha256':sha(__file__)}
    write(out/'summary_public.json',summary); print(json.dumps(summary))

if __name__=='__main__': main()
