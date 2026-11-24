import numpy as np
rng=np.random.default_rng(9); p=211; N=200; out=[]
for t in range(80):
    u=rng.integers(0,p,N); v=rng.integers(0,p,N)
    S=(u[:,None]+v[:,None]*np.arange(p)[None,:])%p
    priv=np.zeros((N,p),bool)
    for k in range(p):
        c=np.bincount(S[:,k],minlength=p); priv[:,k]=c[S[:,k]]==1
    cum=np.cumsum(priv,1)
    r=np.where(cum[:,-1]>=6,np.argmax(cum>=6,1)+1,999); r=np.maximum(r,12); out.append(r)
x=np.concatenate(x:=out) if False else np.concatenate(out)
x=x[x<999]
print("rounds to >=6 private (min 12):",np.median(x),np.percentile(x,99),"-> hours",np.median(x)*p*5.5/3600,np.percentile(x,99)*p*5.5/3600)
# energy budget
E0=0.05*3500*3.85*0.75
print("E0 usable mWh",E0)
J2mWh=1/3.6
Pfl=1.8+1.5*0.05+0.03
A={'floor':Pfl*72,'UI':7.5*J2mWh,'false wakes':0.5*72*0.6*J2mWh,
   'answer rounds(150)':150*(0.015+0.08+0.45+0.03)*J2mWh,'deep tone x2':2*600*0.15*J2mWh,'tap cues(100)':100*(0.009+0.08)*J2mWh}
tot=sum(A.values()); print({k:round(v,1) for k,v in A.items()},'total',round(tot,1),'reserve',round(E0-tot,1),'life h',72+(E0-tot)/Pfl)
ev=tot-A['floor']; print("critical floor mW",(E0-ev)/72)
Bd={'off drain':0.3*72,'boots 12x':12*15,'windows':12*0.5*7,'2 sessions':2*(20*0.575*J2mWh+20*19.34/60*7),'UI':7.5*J2mWh}
print({k:round(v,1) for k,v in Bd.items()},'total',round(sum(Bd.values()),1))
print("tap/LRA energy-equivalence: 1 tap = 1400 s LRA =",1400*0.15*J2mWh,"mWh")
