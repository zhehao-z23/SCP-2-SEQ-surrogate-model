"""Sharp allocation bounds for threshold mass from finite bins and open tails.

No midpoint or width is assigned to an open bin. Query endpoints must not be
bin edges; this avoids conflating the producer's endpoint atom conventions.
"""
import itertools
import numpy as np


def allocation_categories(edges, numerator_interval, conditioning_interval):
    e = np.asarray(edges, dtype=float)
    assert e.ndim == 1 and np.isfinite(e).all() and (np.diff(e)>0).all()
    nlo,nhi = numerator_interval; dlo,dhi = conditioning_interval
    assert dlo <= nlo < nhi <= dhi
    cuts = [x for x in [nlo,nhi,dlo,dhi] if np.isfinite(x)]
    assert all(not np.any(e == x) for x in cuts), 'Endpoint atom rules need a separate aligned-boundary implementation'
    bounds = np.r_[-np.inf,e,np.inf]; categories=[]
    def overlap(lo,hi,a,b): return min(hi,b)>max(lo,a)
    for lo,hi in zip(bounds[:-1],bounds[1:]):
        allowed=[]
        if overlap(lo,hi,nlo,nhi): allowed.append('N')
        if overlap(lo,hi,dlo,nlo) or overlap(lo,hi,nhi,dhi): allowed.append('R')
        if overlap(lo,hi,-np.inf,dlo) or overlap(lo,hi,dhi,np.inf): allowed.append('O')
        assert allowed
        categories.append(tuple(allowed))
    return categories


def fraction_bounds(p, edges, numerator_interval, conditioning_interval=(-np.inf,np.inf)):
    p = np.asarray(p,float)
    assert p.ndim==2 and p.shape[1]==len(edges)+1 and np.isfinite(p).all() and (p>=0).all()
    assert (p.sum(1)>0).all()
    categories=allocation_categories(edges,numerator_interval,conditioning_interval)
    guaranteed_D=p[:,[i for i,c in enumerate(categories) if 'O' not in c]].sum(1)
    assert (guaranteed_D>0).all(), 'Conditioning denominator is not certified positive'
    # A linear-fractional function on this allocation polytope has extrema at
    # vertices. Moving any mass R->O->N can only increase N/(N+R).
    result=[]
    for preference in [('R','O','N'),('N','O','R')]:
        selected=[next(x for x in preference if x in c) for c in categories]
        n=p[:,[i for i,x in enumerate(selected) if x=='N']].sum(1)
        r=p[:,[i for i,x in enumerate(selected) if x=='R']].sum(1)
        result.append(n/(n+r))
    lower,upper=result
    assert ((lower>=-1e-14)&(upper<=1+1e-14)&(lower<=upper+1e-14)).all()
    return lower,upper


def selftest():
    rng=np.random.default_rng(1701); tests=0
    for _ in range(100):
        edges=np.array([-8.,-3.,1.,6.])
        p=rng.dirichlet(np.ones(5))[None,:]
        for num,den in [((-np.inf,-5.),(-np.inf,np.inf)),((5.,np.inf),(-np.inf,np.inf)),((-7.,-5.),(-7.,4.)),((2.,4.),(-7.,4.))]:
            lo,hi=fraction_bounds(p,edges,num,den); cat=allocation_categories(edges,num,den)
            allvalues=[]
            for chosen in itertools.product(*cat):
                n=sum(p[0,i] for i,x in enumerate(chosen) if x=='N'); r=sum(p[0,i] for i,x in enumerate(chosen) if x=='R')
                assert n+r>0
                allvalues.append(n/(n+r))
            assert np.isclose(lo[0],min(allvalues),rtol=0,atol=1e-14) and np.isclose(hi[0],max(allvalues),rtol=0,atol=1e-14)
            tests+=1
    # Explicit open-tail conditional uncertainty remains, rather than giving it a center.
    lo,hi=fraction_bounds(np.array([[.2,.3,.2,.2,.1]]),[-8.,-3.,1.,6.],(-7.,-5.),(-7.,4.))
    assert lo[0]<hi[0]
    return {'status':'passed','brute_force_polytope_vertex_cases':tests,'real_labels_read':0,'open_tail_centers_assigned':0}


if __name__=='__main__':
    import json
    print(json.dumps(selftest()))
