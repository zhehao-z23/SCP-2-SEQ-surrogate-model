"""Branch-only clearance-certified local orientation probes; no geometry changes.

The fixed1e-4 probe can cross a thinner valid component. A complete candidate
bound proves the global distance to every other triangle; epsilon is uniformly
min(1e-4,d_other/4), never tuned against potential or probe outcome. Every original
probe is reclassified. All changed-epsilon/original-problem probes additionally
receive untruncated full-mesh winding verification. Original failure is retained.
"""
from pathlib import Path
import argparse,json,time,sys
import numpy as np
from scipy.spatial import cKDTree
import bridge_physical_run as b
from geometry_classifier import GeometryClassifier,_distance_candidates
from orientation_winding_fallback import direct_winding_distance

EPS0=1e-4;BOUNDARY_TOL=1e-8;MIN_EPS=8*BOUNDARY_TOL;ROUND_PAD=1e-10
def audit(mesh,out,old_records):
    mesh=Path(mesh);out=Path(out);out.mkdir(parents=True,exist_ok=True);v=np.load(mesh/'vertices.npy');faces=np.load(mesh/'faces.npy');comp=np.load(mesh/'component_ids.npy');qc=b.load(mesh/'topology_qc.json')
    assert qc['closed']and qc['edge_manifold']and qc['vertex_manifold']and qc['consistent_face_orientation']and qc['all_components_retained']
    facecomp=comp[faces[:,0]]if len(comp)==len(v)else comp;chosen=[]
    for label in np.unique(facecomp):
        ids=np.flatnonzero(facecomp==label);chosen.extend(ids[np.linspace(0,len(ids)-1,min(128,len(ids)),dtype=int)])
    chosen=np.array(chosen);prior=[]
    for path in old_records:
        old=b.load(path);assert old['tested_face_ids']==chosen.tolist();assert old['vertices_sha256']==b.sha(mesh/'vertices.npy')and old['faces_sha256']==b.sha(mesh/'faces.npy');prior.append(old)
    old_bad=set()
    for old in prior:old_bad.update(i for i,(o,n)in enumerate(zip(old['outside_labels'],old['inside_labels']))if o!=0 or n!=1)
    centers=np.empty((len(faces),3));rmax=0.
    for st in range(0,len(faces),200000):
        tri=v[faces[st:st+200000]];cc=tri.mean(axis=1);centers[st:st+len(cc)]=cc;rmax=max(rmax,float(np.linalg.norm(tri-cc[:,None,:],axis=2).max()))
    tree=cKDTree(centers);p=centers[chosen];tri=v[faces[chosen]];fn=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);fn/=np.linalg.norm(fn,axis=1)[:,None];eps=[];clearances=[]
    for i,(point,fid)in enumerate(zip(p,chosen)):
        _,seed=tree.query(point,k=2,workers=1);seed=next(int(z)for z in np.atleast_1d(seed)if z!=fid)
        upper,_=_distance_candidates(point[None,:],np.ascontiguousarray(v[faces[[seed]]]),np.array([0,1]),np.array([0]));upper=float(upper[0]);radius=np.nextafter(upper+rmax+ROUND_PAD,np.inf)
        ids=np.array([z for z in tree.query_ball_point(point,radius,workers=1)if z!=fid],dtype=np.int64)
        if not len(ids):raise RuntimeError('complete nearest-other-face candidates empty')
        distance,which=_distance_candidates(point[None,:],np.ascontiguousarray(v[faces[ids]]),np.array([0,len(ids)]),np.arange(len(ids)));distance=float(distance[0]);epsilon=min(EPS0,distance/4)
        if not np.isfinite(distance)or epsilon<MIN_EPS:raise RuntimeError('geometric clearance below fixed resolvable probe floor')
        eps.append(epsilon);clearances.append({'probe_index':i,'face_id':int(fid),'component':int(facecomp[fid]),'other_face_id':int(ids[int(which[0])]),'distance_other_A':distance,'seed_upper_A':upper,'global_triangle_centroid_radius_A':rmax,'candidate_radius_A':radius,'candidate_count':len(ids),'epsilon_A':epsilon,'changed_from_fixed':epsilon<EPS0})
    eps=np.array(eps);classifier=GeometryClassifier(v,faces);outside=classifier.classify(p+eps[:,None]*fn)['mesh_parity_region'];inside=classifier.classify(p-eps[:,None]*fn)['mesh_parity_region'];checks=[]
    if np.any(outside!=0)or np.any(inside!=1):
        b.write(out/'parity_failure.json',{'outside':outside.tolist(),'inside':inside.tolist(),'clearances':clearances});raise RuntimeError('clearance-certified whole-mesh parity still contradictory; no winding override')
    special=sorted(old_bad|set(np.flatnonzero(eps<EPS0).tolist()))
    for i in special:
        for sign,expected in [(1,0),(-1,1)]:
            result=direct_winding_distance(v,faces,p[i]+sign*eps[i]*fn[i]);result.update(probe_index=i,face_id=int(chosen[i]),side=sign,epsilon_A=float(eps[i]),expected=expected)
            if result['region']!=expected:raise RuntimeError('full-mesh winding contradicts certified parity')
            checks.append(result)
        if i in old_bad:
            for sign in [1,-1]:
                result=direct_winding_distance(v,faces,p[i]+sign*EPS0*fn[i]);result.update(probe_index=i,face_id=int(chosen[i]),side=sign,epsilon_A=EPS0,old_failure_diagnostic_only=True);checks.append(result)
        print('WINDING_CONFIRMED',mesh.name,i,len(special),flush=True)
    # Same corner-major accumulation as frozen checked_surface_normals. Chunking
    # does not reorder any addition; require exact equality against old arrays.
    normals=np.zeros_like(v)
    for corner in range(3):
        for st in range(0,len(faces),200000):
            ff=faces[st:st+200000];tri=v[ff];cr=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);np.add.at(normals,ff[:,corner],cr)
    norms=np.linalg.norm(normals,axis=1)
    if np.any(norms<=0):raise RuntimeError('zero coherent normal')
    normals/=norms[:,None];equal=[]
    for record in old_records:
        normal=Path(record).parent/'surface_normals.npy'
        if normal.exists():
            exact=bool(np.array_equal(normals,np.load(normal)));equal.append({'path':str(normal),'sha256':b.sha(normal),'bitwise_equal':exact})
            if not exact:raise RuntimeError('coherent normal changed; PB sampling cannot be silently reused')
    np.save(out/'surface_normals.npy',normals)
    report={'schema':'clearance_certified_orientation_v3_branch','pass':True,'scope':'orientation-check refinement only; no mesh/PQR/area/normal alteration, no self-intersection certificate','binding':{x:b.ref(mesh/x)for x in ['vertices.npy','faces.npy','vertex_area_weights.npy','component_ids.npy','surface_manifest.json','topology_qc.json']},'original_orientation_records':[b.ref(x)for x in old_records],'fixed_rule':{'epsilon_A':'min(1e-4,global_distance_to_every_other_triangle/4)','floor_A':MIN_EPS,'boundary_tol_A':BOUNDARY_TOL,'candidate_roundoff_pad_A':ROUND_PAD,'full_triangle_candidate_proof':'any triangle nearer than d0 has centroid within d0+rmax; rmax covers all triangles; no candidate cap','not_potential_dependent':True},'all_original_probes':len(chosen),'old_problem_probes':sorted(old_bad),'changed_epsilon_count':len(np.flatnonzero(eps<EPS0)),'whole_mesh_parity_outside':outside.tolist(),'whole_mesh_parity_inside':inside.tolist(),'clearances':clearances,'full_triangle_winding_checks':checks,'unchanged_existing_normals':equal,'new_coherent_normals':b.ref(out/'surface_normals.npy'),'code':b.ref(__file__),'epoch':time.time()};b.write(out/'certificate.json',report);return report
def main():
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);a=p.parse_args();root=b.R/'cases'/a.case;groups={}
    for variant in ['base','refined','shifted','rotation','expanded','parent_refined','overlap12','surface16']:
        d=root/variant;record=d/'geometry/surface_orientation.json'
        if not record.exists():continue
        mesh=d/('surface16'if variant=='surface16'else 'surface8')if variant in ['rotation','surface16']else root/'surface8'
        groups.setdefault(str(mesh),[]).append(record)
    reports=[]
    for i,(mesh,records)in enumerate(groups.items()):
        out=b.R/'records/geometry_v3'/a.case/('mesh_'+str(i));r=audit(mesh,out,records);reports.append({'mesh':mesh,'certificate':b.ref(out/'certificate.json'),'pass':r['pass'],'probes':r['all_original_probes'],'changed':r['changed_epsilon_count'],'original_problem':r['old_problem_probes']});b.write(b.R/'records/geometry_v3'/a.case/'progress.json',reports);print('MESH_CERTIFIED',i,reports[-1],flush=True)
    b.write(b.R/'records/geometry_v3'/a.case/'outcome.json',{'all_meshes_passed':True,'rows':reports,'geometry_changed':False,'normal_values_changed':False,'new_PB_calls':0})
if __name__=='__main__':main()
