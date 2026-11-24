import numpy as np
from scipy.optimize import least_squares
rng=np.random.default_rng(4)
beta_true=1.5; tort=1.3
def model(X,a,b,R):
    l=tort*np.linalg.norm(R-X,axis=1); return a-b*l-np.log(l)
errs=[];errs_h=[];errs_z=[]
for t in range(2000):
    X=np.array([rng.uniform(-2,2),rng.uniform(-2,2),-rng.uniform(1.5,4)])
    gx,gy=np.meshgrid(np.linspace(-3,3,3),np.linspace(-3,3,3))
    R=np.c_[gx.ravel(),gy.ravel(),np.zeros(9)]
    lnE=model(X,0,beta_true,R)+rng.normal(0,4/4.343,9)      # 4 dB heterogeneity
    S=R[np.argmax(lnE)]                                     # loudest strike point
    lt=tort*np.linalg.norm(S-X); l_disp=lt*(1+rng.normal(0,0.15))+rng.normal(0,0.4)
    z_baro=X[2]+rng.normal(0,1.5)
    def res(q):
        x=q[:3];a=q[3];b=q[4]
        r1=(lnE-model(x,a,b,R))/(4/4.343)
        r2=(tort*np.linalg.norm(S-x)-l_disp)/(0.4+0.15*l_disp)
        r3=(x[2]-z_baro)/1.5
        return np.r_[r1,r2,r3]
    q0=np.r_[S[0],S[1],-2.5,lnE.max(),1.0]
    sol=least_squares(res,q0,bounds=([-6,-6,-10,-50,0.3],[6,6,-0.3,50,4]))
    e=sol.x[:3]-X; errs.append(np.linalg.norm(e)); errs_h.append(np.linalg.norm(e[:2])); errs_z.append(abs(e[2]))
print("3D error median/90%:",np.median(errs),np.percentile(errs,90))
print("horizontal median/90%:",np.median(errs_h),np.percentile(errs_h,90))
print("vertical median/90%:",np.median(errs_z),np.percentile(errs_z,90))
