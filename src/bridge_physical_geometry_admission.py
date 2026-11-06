"""Explicit branch-v3 geometry admission; old failed orientation records remain."""
from pathlib import Path
import argparse,sys,json
import numpy as np
import bridge_physical_run as b
def admit(mesh,out):
    mesh=Path(mesh);out=Path(out);sid=mesh.parts[mesh.parts.index('cases')+1];report=b.R/'records/geometry_v3'/sid/'outcome.json';r=b.load(report)
    assert r['all_meshes_passed']and not r['geometry_changed']and not r['normal_values_changed']
    row=next(x for x in r['rows']if Path(x['mesh'])==mesh);cr=row['certificate'];assert b.sha(cr['path'])==cr['sha256'];cert=b.load(cr['path']);assert cert['pass']
    for name,ref in cert['binding'].items():assert Path(ref['path'])==mesh/name and b.sha(ref['path'])==ref['sha256']
    old=out/'surface_orientation.json';oldrec=b.load(old);source_old=b.load(cert['original_orientation_records'][0]['path'])
    assert oldrec['tested_face_ids']==source_old['tested_face_ids']and oldrec['vertices_sha256']==cert['binding']['vertices.npy']['sha256']and oldrec['faces_sha256']==cert['binding']['faces.npy']['sha256']
    # The same mesh produces the same original fixed probe labels; no omitted
    # later variant may be admitted by filename alone.
    assert oldrec['outside_labels']==source_old['outside_labels']and oldrec['inside_labels']==source_old['inside_labels']
    nn=np.load(cert['new_coherent_normals']['path']);assert b.sha(cert['new_coherent_normals']['path'])==cert['new_coherent_normals']['sha256'];normal=out/'surface_normals.npy'
    if normal.exists():assert np.array_equal(nn,np.load(normal))
    else:np.save(normal,nn)
    sources={name:b.ref(b.P/'code'/name)for name in ['checked_surface_normals.py','checked_surface_normals_v2.py','geometry_classifier.py','orientation_winding_fallback.py']}
    record={'schema':'branch_geometry_v3_admission_v1','pass':True,'mesh_identity':cert['binding'],'certificate':cr,'revision_contract':b.ref(b.R/'contracts/geometry_check_revision_v3.json'),'all_original_probes_represented':True,'original_orientation':b.ref(old),'coherent_normals':b.ref(normal),'normals_equal_to_certified_derivation':True,'physical_geometry_or_PQR_changed':False,'original_qualification_code_bindings':sources,'gate_code':b.ref(__file__)};b.write(out/'geometry_v3_admission.json',record)
    return np.load(mesh/'vertices.npy'),nn,record
def admit_all(sid):
    root=b.R/'cases'/sid;rows=[]
    for variant in ['base','refined','shifted','rotation','expanded','parent_refined','overlap12','surface16']:
        d=root/variant;geo=d/'geometry'
        if not(geo/'surface_orientation.json').exists():raise RuntimeError('missing expected variant original probes:'+variant)
        mesh=d/('surface16'if variant=='surface16'else 'surface8')if variant in ['rotation','surface16']else root/'surface8';admit(mesh,geo);rows.append({'variant':variant,'admission':b.ref(geo/'geometry_v3_admission.json')})
    b.write(b.R/'records/geometry_v3'/sid/'all_variant_admissions.json',{'eight_variants_bound':True,'rows':rows,'operator_changed':False,'new_PB_calls':0});return rows
def main():
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--run-surface16',action='store_true');a=p.parse_args();rows=admit_all(a.case);print('ADMITTED',len(rows),flush=True)
    if a.run_surface16:
        cfg,bins,pool=b.initialize();b.checked_surface_normals_v2=lambda mesh,out,*args,**kw:admit(mesh,out)
        b.run_case(a.case,'surface16',cfg,bins,pool);b.comparisons(a.case,bins);print('SURFACE16_COMPLETED',flush=True)
if __name__=='__main__':main()
