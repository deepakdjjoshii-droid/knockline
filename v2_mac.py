import numpy as np
rng=np.random.default_rng(21)
p=211; Tm=5.5; a=0.25; kap=0.30
def run(N, rho_half, eps0, aligned=True, trials=6):
    worst_pair=0; min_priv=10**9; dec=[]; worst_nonpar=0
    for t in range(trials):
        u=rng.integers(0,p,N); v=rng.integers(0,p,N)
        rho=rng.uniform(-rho_half,rho_half,N)
        th=rng.uniform(-eps0/2,eps0/2,N) if aligned else rng.uniform(0,p*p*Tm,N)
        r=np.arange(p); G=r[None,:]*p+(u[:,None]+v[:,None]*r[None,:])%p   # N x p global slot
        T=(G*Tm+a)
        tt=th[:,None]+(1+rho[:,None])*T
        if not aligned: tt=np.mod(tt,p*p*Tm)
        flat=tt.ravel(); owner=np.repeat(np.arange(N),p); rnd=np.tile(r,N)
        o=np.argsort(flat); fs=flat[o]; ow=owner[o]; rr=rnd[o]
        destroyed=np.zeros((N,p),bool); pair={}
        L=len(fs)
        for i in range(L):
            j=i+1
            while j<L and fs[j]-fs[i]<kap:
                if ow[j]!=ow[i]:
                    destroyed[ow[i],rr[i]]=True; destroyed[ow[j],rr[j]]=True
                    key=(min(ow[i],ow[j]),max(ow[i],ow[j])); pair[key]=pair.get(key,0)+1
                j+=1
        keys=u*p+v; uq,cnt=np.unique(keys,return_counts=True); dup=set(uq[cnt>1])
        for (i,j),c in pair.items():
            if keys[i] in dup and keys[i]==keys[j]: continue
            worst_pair=max(worst_pair,c)
            if v[i]!=v[j]: worst_nonpar=max(worst_nonpar,c)
        ok=[i for i in range(N) if keys[i] not in dup]
        priv=(~destroyed).sum(1)
        min_priv=min(min_priv,priv[ok].min())
        cum=np.cumsum(~destroyed,1)
        dec.append(np.where(cum[:,-1]>=6,np.argmax(cum>=6,1)+1,999)[ok])
    d=np.concatenate(dec)
    return worst_pair,worst_nonpar,min_priv,np.median(d),np.percentile(d,99)
cases=[("call-aligned, 20 ppm relative",200,10e-6,0.15,True),
       ("call-aligned, 40 ppm relative (no pilot)",200,20e-6,0.15,True),
       ("call-aligned, 3 ppm relative (after pilot)",200,1.5e-6,0.15,True),
       ("no alignment at all, 20 ppm relative",200,10e-6,0.15,False)]
for name,N,rh,e0,al in cases:
    wp,wn,mp,med,p99=run(N,rh,e0,al)
    print(f"{name}: max destructive/pair={wp} (non-parallel {wn}), min clean pulses/cycle={mp} (bound {p-(N-1)}), rounds to 6 clean: median {med:.0f}, 99% {p99:.0f}")
