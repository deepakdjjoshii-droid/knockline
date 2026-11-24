import numpy as np
from math import comb, log, log10, sqrt, pi, exp
E=30e9; nu=0.2; rho=2400; h=0.20
B=E*h**3/(12*(1-nu**2)); m=rho*h
kap=(B/m)**0.25
cB=lambda f: kap*np.sqrt(2*pi*f); cg=lambda f: 2*cB(f)
print(f"kappa={kap:.2f} m/s^0.5; phase c_B 100/170/300 Hz = {cB(100):.0f}/{cB(170):.0f}/{cB(300):.0f}; group = {cg(100):.0f}/{cg(170):.0f}/{cg(300):.0f} m/s")
for d in [3,5,10]:
    l=1.3*d
    bb=l/cg(100)-l/cg(300)
    tg=l/cg(170); nb=tg*7.0/(2*170)    # spread across 7 Hz FSK band
    print(f"d={d} m (l={l:.1f}): broadband 100-300 Hz dispersion spread {1e3*bb:.2f} ms | within 7 Hz motor band {1e3*nb:.3f} ms")
print("ruler: ms per metre (100 vs 300 Hz):",1e3*(1/cg(100)-1/cg(300)))
# LRA mechanical response
def loss(df,f0,Q): x=Q*((f0+df)/f0-f0/(f0+df)); return 10*log10(1+x*x)
for f0,Q in [(170,10),(150,20),(240,5)]:
    print(f"f0={f0} Q={Q}: B3={f0/Q:.1f} Hz, tau_m={1e3*Q/(pi*f0):.0f} ms, edge loss at +/-2.8 Hz = {loss(2.8,f0,Q):.2f}/{loss(-2.8,f0,Q):.2f} dB, 3*tau_m={3e3*Q/(pi*f0):.0f} ms")
# noncoherent M-FSK digit error (exact)
def Pd(M,EsN0):
    return sum((-1)**(k+1)*comb(M-1,k)/(k+1)*exp(-k/(k+1)*EsN0) for k in range(1,M))
M=15
# find gamma (pulse matched-filter SNR) for P_d targets; E_digit/N0 = gamma/4
for tgt in [0.05,0.02,0.01]:
    g=1.0
    while Pd(M,g/4)>tgt: g*=1.01
    print(f"P_d<={tgt}: gamma_req={10*log10(g):.1f} dB (E_digit/N0={10*log10(g/4):.1f} dB)")
# decode failure with n private, k=4, up to t=(n-4)//2 symbol errors
def Pfail(n,ps,k=4):
    t=(n-k)//2
    return sum(comb(n,e)*ps**e*(1-ps)**(n-e) for e in range(t+1,n+1))
for pdv in [0.05,0.1]:
    ps=1-(1-pdv)**2
    print(f"P_d={pdv}: P_sym={ps:.3f}; P_fail n=12: {Pfail(12,ps):.2e}, n=20: {Pfail(20,ps):.2e}")
# SNR vs depth for new 5 s FSK pulse (tau_eff 4.7 s, edge loss 0.45 dB), using v1 2-s table
base={1:27.9,2:18.9,3:11.2,4:3.9,5:-3.0,6:-9.8,8:-23.0}
for d,g2 in base.items():
    g=g2+10*log10(4.7/2)-0.45
    pdv=Pd(M,10**(g/10)/4)
    print(f"d={d} m: gamma={g:.1f} dB, P_d={min(pdv,1):.3g}")
# bandwidth-limited capacity (W=17 Hz) vs wideband limit, from v1 P/N0 values
for d,C in {1:221,2:28,3:4.72,4:0.894,6:0.0381}.items():
    PN=C*log(2); W=17
    print(f"d={d}: wideband {C:.3g} bit/s -> motor-band-limited {W*np.log2(1+PN/W):.3g} bit/s")
# frequency pilot CRB: sigma_f = sqrt(12)/(2 pi T sqrt(gamma_p)); gamma_p = gamma_up*(Tp/tau_eff)/K
for gup_db in [16,0]:
    gp=10**(gup_db/10)*(600/4.7)/4
    sf=sqrt(12)/(2*pi*600*sqrt(gp))
    print(f"uplink gamma {gup_db} dB -> pilot gamma {10*log10(gp):.1f} dB, sigma_f={sf:.2e} Hz = {sf/170*1e6:.2f} ppm")
# drift condition
p=211; Tm=5.5; eps0=0.15; kap_t=0.30
S=lambda r: r*p*p
print("S(20ppm)=",S(20e-6)," threshold rho <", (1-(eps0+kap_t)/Tm)/p**2*1e6,"ppm; cycle h=",p*p*Tm/3600)
print("tuning-fork crystal at 5 C vs turnover 20/30 C:",-0.034*(5-20)**2,-0.034*(5-30)**2,"ppm")
