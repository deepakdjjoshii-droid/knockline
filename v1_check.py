import numpy as np
rng=np.random.default_rng(5); p=211
# 1) deterministic bound excluding phones whose line is duplicated
worst=10**9
for t in range(60):
    N=200; u=rng.integers(0,p,N); v=rng.integers(0,p,N)
    keys=u*p+v; uniq,cnt=np.unique(keys,return_counts=True); dupk=set(uniq[cnt>1])
    S=(u[:,None]+v[:,None]*np.arange(p)[None,:])%p
    priv=np.zeros((N,p),bool)
    for k in range(p):
        c=np.bincount(S[:,k],minlength=p); priv[:,k]=c[S[:,k]]==1
    ok=[i for i in range(N) if keys[i] not in dupk]
    worst=min(worst,priv[ok].sum(1).min())
print("min private per cycle over 60x200 phones (distinct lines):",worst,"bound",p-199)
# 2) async pair collisions, fixed counting
N=80; u=rng.integers(0,p,N); v=rng.integers(0,p,N); off=rng.uniform(0,p*p,N)
r=np.arange(p)
g=[r*p+(u[i]+v[i]*r)%p+off[i] for i in range(N)]
mx=0; hist={}
for i in range(N):
    for j in range(N):
        if i==j or v[i]==v[j]: continue
        gj=np.concatenate([g[j]-p*p,g[j],g[j]+p*p])
        c=int((np.abs(g[i][:,None]-gj[None,:])<1).sum()); mx=max(mx,c); hist[c]=hist.get(c,0)+1
print("async max collisions per pair per cycle:",mx,"histogram:",dict(sorted(hist.items())))
