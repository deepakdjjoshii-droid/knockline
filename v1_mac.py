import numpy as np
rng=np.random.default_rng(11)
p=211
def sim_sync(N,W,trials=200,need=5):
    # returns: min private over phones per full cycle check, decode-round distribution
    dec_rounds=[]; minpriv=[]; dup=0
    for t in range(trials):
        u=rng.integers(0,p,N); v=rng.integers(0,p,N)
        lines=set(zip(u.tolist(),v.tolist()))
        if len(lines)<N: dup+=1
        r=np.arange(W)
        S=(u[:,None]+v[:,None]*r[None,:])%p          # N x W slots
        priv=np.zeros((N,W),bool)
        for k in range(W):
            col=S[:,k]; cnt=np.bincount(col,minlength=p)
            priv[:,k]=cnt[col]==1
        cum=np.cumsum(priv,axis=1)
        reach=np.where(cum[:,-1]>=need, np.argmax(cum>=need,axis=1)+1, W+1)
        dec_rounds.append(reach)
        if W>=p: minpriv.append(priv[:,:p].sum(1).min())
    return np.concatenate(dec_rounds),dup/trials,minpriv
for N in [20,50,200]:
    d,dup,mp=sim_sync(N,p,trials=60 if N==200 else 100)
    print(f"N={N}: P(private/round)={(1-1/p)**(N-1):.3f}  rounds to >=5 private: median {np.median(d):.0f}, 99% {np.percentile(d,99):.0f}, max {d.max()}  | min private per cycle {min(mp)} (bound {p-(N-1)})  dup-line trials {dup:.2f}")
# identity ghosts in OR channel (no fingerprint): expected ghosts after W rounds
for N in [20,50,200]:
    occ=1-(1-1/p)**N
    for W in [4,6,10,20,30]:
        pass
    Ws=[W for W in range(2,60) if p*p*occ**W<0.01]
    print(f"N={N}: slot occupancy {occ:.3f}; rounds until expected ghost lines <0.01: {Ws[0]}")
# verify ghost count by direct Hough on one trial N=200
N=200; u=rng.integers(0,p,N); v=rng.integers(0,p,N)
true=set(zip(u.tolist(),v.tolist()))
for W in [10,20,30]:
    occ=np.zeros((W,p),bool)
    for r in range(W): occ[r,(u+v*r)%p]=True
    A=np.arange(p)[:,None]; Bv=np.arange(p)[None,:]
    ok=np.ones((p,p),bool)
    for r in range(W): ok&=occ[r,(A+Bv*r)%p]
    found=set(zip(*np.nonzero(ok)))
    print(f"Hough N=200 W={W}: accepted {len(found)}, true found {len(true&found)}/{len(true)}, ghosts {len(found-true)}")
# asynchronous: arbitrary real offsets, max collisions per pair per cycle
N=60; u=rng.integers(0,p,N); v=rng.integers(0,p,N); off=rng.uniform(0,p*p,N)
starts=[]
for i in range(N):
    r=np.arange(p+2); g=r*p+(u[i]+v[i]*r)%p+off[i]; starts.append(g)
mx=0
for i in range(N):
    gi=np.arange(p)*p+(u[i]+v[i]*np.arange(p))%p+off[i]
    for j in range(N):
        if i==j or v[i]==v[j]: continue
        gj=np.concatenate([starts[j]-p*p,starts[j],starts[j]+p*p])
        c=(np.abs(gi[:,None]-gj[None,:])<1).sum()
        mx=max(mx,c)
print("async: max collisions between any pair (distinct slopes) in one cycle:",mx)
print("P(phone shares its exact line with someone, N=200):",1-(1-1/p**2)**199)
print("async expected private fraction N=200:",np.exp(-2*199/p))
