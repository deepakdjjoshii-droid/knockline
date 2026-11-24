import numpy as np
from math import comb
np.random.seed(7)
# ---------- slab / wave physics ----------
E=30e9; nu=0.2; rho=2400; h=0.20
B=E*h**3/(12*(1-nu**2)); m=rho*h
f=170.0; w=2*np.pi*f
kB=(w**2*m/B)**0.25; cB=w/kB; lamB=cB/f
Y=1/(8*np.sqrt(B*m))
c0=343; rho0=1.2
fc=c0**2/(2*np.pi)*np.sqrt(m/B)
Q=10; alpha=kB/(4*Q)
LJ=8.0; hL=1.5; tort=1.3
print(f"B={B:.3g} N*m  m''={m:.0f} kg/m2  kB={kB:.3f}/m  cB={cB:.0f} m/s  lambda={lamB:.2f} m")
print(f"Y={Y:.3g} m/(N s)  f_coinc={fc:.0f} Hz  alpha={alpha:.4f} Np/m = {8.686*alpha:.2f} dB/m")
def G(d):
    l=tort*d
    return np.sqrt(2/(np.pi*kB*l))*np.exp(-alpha*l)*10**(-(d/hL)*LJ/20)
# ---------- receivers ----------
radfac=np.sqrt(1-fc/f)
n_mic=2e-5                     # Pa/rtHz at 170 Hz, mic in rubble crevice, silence period
nv_mic=n_mic*radfac/(rho0*c0)
nv_acc=150e-6*9.81/w           # 150 ug/rtHz MEMS accel
print(f"velocity-equiv noise: mic {nv_mic:.2e}, accel {nv_acc:.2e} m/s/rtHz  (accel worse by {20*np.log10(nv_acc/nv_mic):.1f} dB)")
F0=1.0; P_lra=0.15
K=4; Bw=350.0
I_tap=1.0; I_sledge=25.0
db=lambda x:10*np.log10(x)
print("\nd[m]  G[dB]  LRA2s(K4mic)  LRA600s  tap(K4mic)  sledge->phone mic  sledge->phone acc  C_LRA[bit/s]")
for d in [1,2,3,4,5,6,8,10,12,15]:
    g=G(d); v=F0*Y*g
    s_lra=K*v**2*2/nv_mic**2
    s_lra600=K*v**2*600/nv_mic**2
    s_tap=K*4*(I_tap*Y*g)**2*Bw/nv_mic**2
    s_sl=4*(I_sledge*Y*g)**2*Bw/nv_mic**2
    s_sla=4*(I_sledge*Y*g)**2*Bw/nv_acc**2
    C=K*v**2/(2*nv_mic**2*np.log(2))
    print(f"{d:4.0f} {20*np.log10(g):7.1f} {db(s_lra):10.1f} {db(s_lra600):9.1f} {db(s_tap):10.1f} {db(s_sl):14.1f} {db(s_sla):16.1f}   {C:.3g}")
print("tap-equivalent LRA seconds:",4*I_tap**2*Bw/F0**2)
