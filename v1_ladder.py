import numpy as np
rng=np.random.default_rng(3)
Tm=5.5
stages=[(23,10,8),(53,10,6),(211,10**6,5)]  # (p, rounds, private pulses needed)
def run(N,trials=300):
    times=[]
    for t in range(trials):
        t0=0.0; done=np.full(N,np.nan)
        for (p,R,need) in stages:
            u=rng.integers(0,p,N); v=rng.integers(0,p,N)
            Rr=R if R<10**6 else 211
            S=(u[:,None]+v[:,None]*np.arange(Rr)[None,:])%p
            priv=np.zeros((N,Rr),bool)
            for k in range(Rr):
                c=np.bincount(S[:,k],minlength=p); priv[:,k]=c[S[:,k]]==1
            occ=1-(1-1/p)**N
            cum=np.cumsum(priv,1)
            for i in range(N):
                if not np.isnan(done[i]): continue
                ok=np.nonzero(cum[i]>=need)[0]
                if len(ok)==0: continue
                W=max(ok[0]+1, int(np.ceil(np.log(0.1/p**2)/np.log(occ))) if occ>0 else 1)
                if W<=Rr: done[i]=t0+W*p*Tm/60
            t0+=Rr*p*Tm/60
        times.append(done)
    x=np.concatenate(times)
    return np.nanmedian(x),np.nanpercentile(x,95),np.mean(np.isnan(x))
for N in [3,10,20,50,200]:
    med,p95,fail=run(N, 300 if N<200 else 40)
    print(f"N={N:3d}: median {med:6.0f} min, 95th pct {p95:6.0f} min, undelivered within 68h-cycle {fail:.3%}")
