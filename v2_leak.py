#!/usr/bin/env python3
"""
v2_leak.py -- KNOCKLINE v2 near-far / spectral-leakage Monte Carlo.

Question: how many of 200 phones decode their message within 68 h under real
spectral leakage and near-far effects?

Physical layer (all closed form, checked against brute-force sampling):
  * every motor tone segment [t0,t1) of frequency f is the response of a
    one-pole resonator (tau = Q/(pi f0)) to a gated drive:
        steady  B e^{j2pi f (t-t0)}              on [t0,t1)
        settle -B e^{p (t-t0)}, p=-1/tau+j2pi f0  on [t0,t1)
        ring    y(t1) e^{p (t-t1)}                on [t1,inf)
    |B| = F*Y*|H(f)|, |H|^2 = 1/(1+Q^2(f/f0-f0/f)^2), phase of the one-pole.
  * static echo channel per phone/receiver: Rician (K=3 dB) direct tap + 5
    echo taps, exponential power-delay profile over the spread.
  * receiver bin b of a digit window w(t), t in [0.15,2.5] s of the digit:
        Y_b = sum_sources sum_taps sum_pieces int w(t) Re{piece(t)} e^{-j2pi f_b t} dt
    (analytic part plus the negative-frequency image, so the real signal is exact)
    + correlated complex Gaussian noise (covariance from int w^2 e^{-j2pi df t}).
Digit = argmax over the phone's 15 bins of |sum_m conj(h_m) Y_m|^2, i.e. coherent
combining with the learned (static) complex gain of each receiver; erasure if below a CFAR threshold or invalid symbol.
Message = cubic over F_211 evaluated at r; errors-and-erasures decoding
(correct iff >=6 agreeing symbols and agreements - errors >= 4); wrong
decodes are searched for explicitly (other phones' cubics, and exhaustive
4-point interpolation while <=14 symbols are unerased).

Usage:  python3 v2_leak.py [--seeds 50] [--procs 4] [--only c_full,...]
Outputs go to results/leak/.
"""
import numpy as np, os, sys, time, csv, argparse, itertools, json
from math import comb, exp, ceil
from multiprocessing import Pool
from scipy import stats

# ---------------------------------------------------------------- constants
P = 211; TM = 5.5; AOFF = 0.25; TD = 2.5; TP = 5.0
GRID = 0.4; NB = 15
WS, WE = 0.15, 2.5; LW = WE - WS
NV = 3.34e-8; YM = 1.25e-6
KB = 2.26; ALPHA = 0.0566; TORT = 1.3
CG = 943.0                       # flexural group velocity at 170 Hz, m/s
NTAP = 6
CYCLE = P * P * TM               # 68.0 h
CHECK_H = [6, 12, 24, 68]
RX8 = np.array([[5, 5], [5, 15], [15, 5], [15, 15],
                [10, 2], [10, 18], [2, 10], [18, 10]], float)
HAMMER = np.array([10.0, 10.0])
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'leak')

BASE = dict(N=200, nrx=4, scatter_db=4.0, clust_frac=0.6, clust_hw=2.0,
            echo=(0.02, 0.10), kfac_db=3.0, drift_ppm=10.0, align=0.15,
            mode='full', window='rect',
            pc=False, pc_ref_depth=2.0, pc_min=0.05, pc_meas_db=1.0,
            sic=False, sic_resid_db=-20.0,
            pfa_bin=1e-2, prune_db=-30.0)

CONFIGS = {
    'a_alone':    dict(mode='alone'),
    'b_ideal':    dict(mode='ideal'),
    'c_full':     dict(),
    'm_hann':     dict(window='hann'),
    'm_pc':       dict(pc=True),
    'm_sic':      dict(sic=True),
    'm_8rx':      dict(nrx=8),
    'm_all':      dict(pc=True, sic=True, nrx=8),
    'm_all_hann': dict(pc=True, sic=True, nrx=8, window='hann'),
    's_clust0':   dict(clust_frac=0.0),
    's_clust0_b': dict(clust_frac=0.0, mode='ideal'),
    's_clust90':  dict(clust_frac=0.9),
    's_clust90_b': dict(clust_frac=0.9, mode='ideal'),
    's_echo20':   dict(echo=(0.02, 0.02)),
    's_echo100':  dict(echo=(0.10, 0.10)),
    's_scat2':    dict(scatter_db=2.0),
    's_scat6':    dict(scatter_db=6.0),
    's_drift20':  dict(drift_ppm=20.0),
    's_drift20_b': dict(drift_ppm=20.0, mode='ideal'),
}
LABEL = {
    'a_alone': '(a) each phone alone', 'b_ideal': '(b) collisions, ideal tones',
    'c_full': '(c) full leakage', 'm_hann': '(c)+Hann window',
    'm_pc': '(c)+power control', 'm_sic': '(c)+subtract decoded',
    'm_8rx': '(c)+8 receivers', 'm_all': '(c)+PC+SIC+8rx',
    'm_all_hann': '(c)+PC+SIC+8rx+Hann',
    's_clust0': 'clustering 0%', 's_clust0_b': 'clustering 0% (ideal)',
    's_clust90': 'clustering 90%', 's_clust90_b': 'clustering 90% (ideal)',
    's_echo20': 'echo spread 20 ms', 's_echo100': 'echo spread 100 ms',
    's_scat2': 'scatter 2 dB', 's_scat6': 'scatter 6 dB',
    's_drift20': 'drift +-20 ppm (40 rel)', 's_drift20_b': 'drift +-20 ppm (ideal)',
}

INV = np.zeros(P, np.int64)
INV[1:] = [pow(i, P - 2, P) for i in range(1, P)]
COMBS = {n: np.array(list(itertools.combinations(range(n), 4))) for n in range(4, 15)}


# ---------------------------------------------------------------- windows
def window_terms(name):
    """w(t) on [0,LW] = sum c_i exp(j 2pi nu_i t); returns c, nu, int w^2."""
    if name == 'rect':
        return np.array([1.0 + 0j]), np.array([0.0]), LW
    if name == 'hann':
        return (np.array([0.5, -0.25, -0.25]) + 0j,
                np.array([0.0, 1 / LW, -1 / LW]), 3 * LW / 8)
    raise ValueError(name)


def w_of_t(name, t):
    if name == 'rect':
        return np.ones_like(t)
    return 0.5 - 0.5 * np.cos(2 * np.pi * t / LW)


_CHOL = {}
def noise_chol(name):
    if name not in _CHOL:
        t = np.linspace(0, LW, 8001)
        w2 = w_of_t(name, t) ** 2
        df = GRID * (np.arange(NB)[:, None] - np.arange(NB)[None, :])
        R = np.trapezoid(w2[None, None, :] * np.exp(-2j * np.pi * df[..., None] * t), t, axis=-1)
        R /= R[0, 0].real
        _CHOL[name] = np.linalg.cholesky(R)
    return _CHOL[name]


# ---------------------------------------------------------------- world
def path_gain(slant, depth):
    l = TORT * slant
    return np.sqrt(2 / (np.pi * KB * l)) * np.exp(-ALPHA * l) * 10 ** (-(depth / 1.5) * 8 / 20), l


def motor_H(f, f0, Q):
    mag = 1 / np.sqrt(1 + Q ** 2 * (f / f0 - f0 / f) ** 2)
    tau = Q / (np.pi * f0)
    return mag * np.exp(-1j * np.arctan(2 * np.pi * tau * (f - f0)))


def make_world(cfg, seed):
    """All random draws use fixed sub-streams so configs share common random numbers."""
    N = cfg['N']; M = cfg['nrx']
    R = lambda i: np.random.default_rng([seed, i])
    g = R(1); x = g.uniform(0, 20, N); y = g.uniform(0, 20, N); dep = g.uniform(1, 6, N)
    g = R(2); F = g.uniform(0.7, 1.5, N)
    g = R(3)
    nc = g.integers(4, 7); cc = g.uniform(150, 240, 6); cQ = g.uniform(5, 20, 6)
    inc = g.random(N) < cfg['clust_frac']; ci = g.integers(0, nc, N)
    off = g.uniform(-1, 1, N); fu = g.uniform(150, 240, N)
    qj = g.uniform(0.9, 1.1, N); qu = g.uniform(5, 20, N)
    f0 = np.where(inc, cc[ci] + cfg['clust_hw'] * off, fu)
    Q = np.where(inc, cQ[ci] * qj, qu)
    g = R(4)
    u = g.integers(0, P, N); v = g.integers(0, P, N)
    rho = g.uniform(-1, 1, N) * cfg['drift_ppm'] * 1e-6
    th = g.uniform(-0.5, 0.5, N) * cfg['align']
    coef = g.integers(0, P, (N, 4))
    g = R(5)
    z = g.standard_normal((N, 8))
    sp = g.uniform(0, 1, (N, 8))
    tu = np.sort(g.uniform(0, 1, (N, 8, NTAP - 1)), axis=-1)
    tg = (g.standard_normal((N, 8, NTAP - 1)) + 1j * g.standard_normal((N, 8, NTAP - 1))) / np.sqrt(2)
    dph = g.uniform(0, 2 * np.pi, (N, 8))
    g = R(6); phs = g.uniform(0, 2 * np.pi, (N, P, 2))
    g = R(7); zd = g.standard_normal(N); zm = g.standard_normal(N)
    g = R(9); sic_ph = g.uniform(0, 2 * np.pi, (N, P))     # residual phase per source pulse

    # receivers / path gains
    rx = RX8[:M]
    dxy = np.hypot(x[:, None] - rx[None, :, 0], y[:, None] - rx[None, :, 1])
    slant = np.sqrt(dxy ** 2 + dep[:, None] ** 2)
    G, l = path_gain(slant, dep[:, None])
    G = G * 10 ** (cfg['scatter_db'] * z[:, :M] / 20)
    tau_p = l / CG
    # echo channel
    lo_s, hi_s = cfg['echo']
    S_sp = lo_s + (hi_s - lo_s) * sp[:, :M]
    dly = np.concatenate([np.zeros((N, M, 1)), S_sp[..., None] * tu[:, :M]], axis=-1)
    K = 10 ** (cfg['kfac_db'] / 10)
    pw = np.exp(-dly[..., 1:] / (S_sp[..., None] / 3 + 1e-12))
    pw = pw / pw.sum(-1, keepdims=True) / (K + 1)
    gq = np.concatenate([np.sqrt(K / (K + 1)) * np.exp(1j * dph[:, :M, None]),
                         np.sqrt(pw) * tg[:, :M]], axis=-1)            # (N,M,NTAP)
    # power control from downlink level (hammer at footprint centre)
    if cfg['pc']:
        sd = np.sqrt(np.hypot(x - HAMMER[0], y - HAMMER[1]) ** 2 + dep ** 2)
        Gd, _ = path_gain(sd, dep)
        Gd = Gd * 10 ** ((cfg['scatter_db'] * zd + cfg['pc_meas_db'] * zm) / 20)
        Gref, _ = path_gain(cfg['pc_ref_depth'], cfg['pc_ref_depth'])
        F = F * np.clip(Gref / Gd, cfg['pc_min'], 1.0)
    # schedule
    rr = np.arange(P)
    slot = (u[:, None] + v[:, None] * rr[None, :]) % P
    S = th[:, None] + (1 + rho[:, None]) * ((rr[None, :] * P + slot) * TM + AOFF)
    # message
    sym = (coef[:, 0:1] + coef[:, 1:2] * rr + coef[:, 2:3] * rr ** 2 % P
           + coef[:, 3:4] * (rr ** 3 % P)) % P
    # bands
    gc = np.round(f0 / GRID).astype(int)
    gidx = gc[:, None] + np.arange(-7, 8)[None, :]
    fb = GRID * gidx
    dig_of_bin = gidx % 15
    tone_bin = np.argsort(dig_of_bin, axis=1)          # tone_bin[k,d] = bin index carrying digit d
    d1, d2 = sym // 15, sym % 15
    fton = np.stack([np.take_along_axis(fb, tone_bin[np.arange(N)[:, None], d1], 1),
                     np.take_along_axis(fb, tone_bin[np.arange(N)[:, None], d2], 1)], -1)  # (N,P,2)
    # learned combining weights
    Cc = (gq * np.exp(-2j * np.pi * (fb[:, 7])[:, None, None] * dly)).sum(-1)
    # learned complex gain per receiver and bin (static channel) -> coherent (MRC) combining
    Cb = (gq[:, :, None, :] * np.exp(-2j * np.pi * fb[:, None, :, None] * dly[:, :, None, :])).sum(-1)
    hc = G[:, :, None] * Cb                                            # (N,M,NB)
    hc = hc / np.sqrt((np.abs(hc) ** 2).sum(1, keepdims=True))
    Hc = np.abs(motor_H(fb[:, 7], f0, Q))
    gamma = ((F * YM * Hc)[:, None] * G * np.abs(Cc)) ** 2 * (2 * LW) / NV ** 2
    gamma = gamma.sum(1)
    return dict(N=N, M=M, x=x, y=y, dep=dep, F=F, f0=f0, Q=Q, inc=inc, u=u, v=v, rho=rho,
                th=th, coef=coef, G=G, tau=tau_p, dly=dly, gq=gq, S=S, sym=sym, fb=fb,
                dig_of_bin=dig_of_bin, fton=fton, phs=phs, hc=hc, gamma=gamma,
                sic_ph=sic_ph, seed=seed)


# ---------------------------------------------------------------- kernel
def _geo(step):
    """(n,) complex ratio -> (n,NB) powers step**b via cumulative product."""
    out = np.empty((len(step), NB), complex)
    out[:, 0] = 1
    out[:, 1:] = step[:, None]
    return np.cumprod(out, axis=1, out=out)


def kernel(coef, s, anchor, L, U, fb, wc, wnu):
    """sum_i c_i * int_L^U e^{s(t-anchor)} e^{j2pi(nu_i - fb) t} dt, times coef. -> (n,NB)
    fb must be a regular grid fb[:,b] = fb[:,0] + GRID*b (exploited via geometric powers)."""
    D = U - L
    bb = np.arange(NB)
    pwL = _geo(np.exp(-2j * np.pi * GRID * L))
    pwD = _geo(np.exp(-2j * np.pi * GRID * D))
    out = np.zeros(fb.shape, complex)
    for c, nu in zip(wc, wnu):
        om0 = 2j * np.pi * (nu - fb[:, 0])
        pre = np.exp(om0 * L + s * (L - anchor))[:, None] * pwL
        E = np.exp((s + om0) * D)[:, None] * pwD
        sig = (s + om0)[:, None] - 2j * np.pi * GRID * bb[None, :]
        zz = sig * D[:, None]
        small = np.abs(zz) < 1e-6
        ratio = np.where(small, D[:, None] * (1 + zz / 2), (E - 1) / np.where(small, 1.0, sig))
        out += c * pre * ratio
    return coef[:, None] * out


IMG_FLOOR = 1e-6     # image terms provably below this (noise-sigma units, -120 dB) are skipped


def kernel_real(coef, s, anchor, L, U, fb, wc, wnu):
    """Real signal Re{coef e^{s(t-a)}} = analytic part + negative-frequency image.
    |image| <= |coef| sum|c_i| 2/|Im sigma| with |Im sigma| >= 2pi(f_src + fb_min - max nu)."""
    out = kernel(coef, s, anchor, L, U, fb, wc, wnu)
    bound = np.abs(coef) * np.abs(wc).sum() * 2 / (2 * np.pi * (s.imag / (2 * np.pi) + fb[:, 0] - 1.0))
    m = bound > IMG_FLOOR
    if m.any():
        out[m] += kernel(np.conj(coef[m]), np.conj(s[m]), anchor[m], L[m], U[m], fb[m], wc, wnu)
    return out


def pulse_pieces(W, j, rj):
    """Six pieces of source pulse (j,rj) in source-local time (t=0 at pulse start)."""
    n = len(j)
    f0, Q, F = W['f0'][j], W['Q'][j], W['F'][j]
    tau = Q / (np.pi * f0); p = -1 / tau + 2j * np.pi * f0
    coef = np.zeros((n, 6), complex); s = np.zeros((n, 6), complex)
    anc = np.zeros((n, 6)); lo = np.zeros((n, 6)); hi = np.zeros((n, 6))
    ftone = np.zeros((n, 6))
    for d in range(2):
        f = W['fton'][j, rj, d]
        B = F * YM * motor_H(f, f0, Q) * np.exp(1j * W['phs'][j, rj, d])
        t0, t1 = d * TD, (d + 1) * TD
        y1 = B * (np.exp(2j * np.pi * f * TD) - np.exp(p * TD))
        c = 3 * d
        coef[:, c], s[:, c], anc[:, c], lo[:, c], hi[:, c], ftone[:, c] = B, 2j * np.pi * f, t0, t0, t1, f
        coef[:, c + 1], s[:, c + 1], anc[:, c + 1], lo[:, c + 1], hi[:, c + 1] = -B, p, t0, t0, t1
        coef[:, c + 2], s[:, c + 2], anc[:, c + 2], lo[:, c + 2], hi[:, c + 2] = y1, p, t1, t1, np.inf
        ftone[:, c + 1] = ftone[:, c + 2] = np.nan
    return coef, s, anc, lo, hi, ftone, tau


def build_pairs(W, cfg):
    N = W['N']; S = W['S']
    flat = S.ravel(); order = np.argsort(flat); sf = flat[order]
    lo = np.searchsorted(sf, flat - (TP + 0.9)); hi = np.searchsorted(sf, flat + TP + 0.2)
    cnt = hi - lo
    t_idx = np.repeat(np.arange(N * P), cnt)
    pos = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt) + np.repeat(lo, cnt)
    s_idx = order[pos]
    if cfg['mode'] == 'alone':
        keep = s_idx == t_idx
        t_idx, s_idx = t_idx[keep], s_idx[keep]
    own = s_idx == t_idx
    # prune interferers whose leakage bound is far below the noise floor
    k, j = t_idx // P, s_idx // P; rj = s_idx % P
    fbk = W['fb'][k]
    ft = W['fton'][j, rj]                                   # (n,2)
    dt = np.abs(ft[:, :, None] - fbk[:, None, :]).min((1, 2))
    dp = np.abs(W['f0'][j][:, None] - fbk).min(1)
    taus = W['Q'][j] / (np.pi * W['f0'][j])
    sp = np.sqrt(1 / taus ** 2 + (2 * np.pi * dp) ** 2)
    bnd = 2 * 2 / (2 * np.pi * np.maximum(dt, 1e-9)) + 6 * np.minimum(taus, 2 / sp)
    if cfg['window'] == 'hann':
        bnd = bnd * 2
    amp = W['F'][j] * YM * (W['G'][j] * np.abs(W['gq'][j]).sum(-1)).max(1)
    _, _, w2 = window_terms(cfg['window'])
    sig_n = np.sqrt(NV ** 2 / 2 * w2)
    keep = own | (0.5 * amp * bnd > 10 ** (cfg['prune_db'] / 20) * sig_n)
    return t_idx[keep], s_idx[keep], own[keep]


def compute_Y(W, cfg, t_idx, s_idx, own, store_int=False, chunk=3000):
    """Noise-free bin values. Returns Ywin (N*P,2,M,NB) and optionally per-pair interference."""
    N, M = W['N'], W['M']
    wc, wnu, w2 = window_terms(cfg['window'])
    sig_n = np.sqrt(NV ** 2 / 2 * w2)
    ideal = cfg['mode'] == 'ideal'
    tot = N * P * 2 * M * NB
    Yr = np.zeros(tot); Yi = np.zeros(tot)
    ints = [] if store_int else None
    dmax_all = W['dly'].max(-1)                               # (N,M)
    for a in range(0, len(t_idx), chunk):
        ti, si, ow = t_idx[a:a + chunk], s_idx[a:a + chunk], own[a:a + chunk]
        n = len(ti)
        k, r = ti // P, ti % P; j, rj = si // P, si % P
        coef, s, anc, lo, hi, ftone, taus = pulse_pieces(W, j, rj)
        # item axes: (pair, digit, rx, piece)
        Wst = W['S'][k, r][:, None] + np.array([0, TD])[None, :] + WS        # (n,2)
        offs = (W['S'][j, rj][:, None, None] + W['tau'][j][:, None, :]
                - Wst[:, :, None] - W['tau'][k][:, None, :])                  # (n,2,M)
        sh = (n, 2, M, 6)
        o4 = offs[..., None]
        A0 = np.broadcast_to(anc[:, None, None, :], sh) + o4
        L0 = np.broadcast_to(lo[:, None, None, :], sh) + o4
        H0 = np.broadcast_to(hi[:, None, None, :], sh) + o4
        dmx = np.broadcast_to(dmax_all[j][:, None, :, None], sh)
        cf = np.broadcast_to(coef[:, None, None, :], sh) * (0.5 * W['G'][j] / sig_n)[:, None, :, None]
        ss = np.broadcast_to(s[:, None, None, :], sh)
        jj = np.broadcast_to(j[:, None, None, None], sh)
        mm = np.broadcast_to(np.arange(M)[None, None, :, None], sh)
        kk = np.broadcast_to(k[:, None, None, None], sh)
        tau4 = np.broadcast_to(taus[:, None, None, None], sh)
        steady = np.zeros(6, bool); steady[[0, 3]] = True
        st4 = np.broadcast_to(steady[None, None, None, :], sh)
        own4 = np.broadcast_to(ow[:, None, None, None], sh)
        widx = ((np.broadcast_to(ti[:, None, None, None], sh) * 2
                 + np.arange(2)[None, :, None, None]) * M + mm)             # window-rx index
        pidx = np.broadcast_to(np.arange(n)[:, None, None, None] * (2 * M)
                               + np.arange(2)[None, :, None, None] * M + mm, sh)
        ftn = np.broadcast_to(ftone[:, None, None, :], sh)
        none = (H0 + dmx <= 0) | (L0 >= LW) | ((~st4) & (H0 + dmx <= 0)) | \
               ((~st4) & np.isinf(H0) & (L0 < -40 * tau4))
        if ideal:
            none = none | ((~own4) & (~st4))
        cover = (~none) & (L0 + dmx <= 0) & (H0 >= LW)
        strad = (~none) & (~cover)
        if ideal:          # interferers: steady tones, direct timing, channel C(f), exact bin only
            cover = cover | (strad & ~own4)
            strad = strad & own4
        contrib_idx = []; contrib_val = []
        # cover items: tap sum factorises
        ci = np.nonzero(cover.ravel())[0]
        if len(ci):
            jc, mc = jj.ravel()[ci], mm.ravel()[ci]
            sc = ss.ravel()[ci]
            Cs = (W['gq'][jc, mc] * np.exp(-sc[:, None] * W['dly'][jc, mc])).sum(-1)
            Lc = np.zeros(len(ci)); Uc = np.full(len(ci), LW)
            if ideal:
                Lc = np.clip(L0.ravel()[ci], 0, LW); Uc = np.clip(H0.ravel()[ci], 0, LW)
            kf = kernel if ideal else kernel_real
            val = kf(cf.ravel()[ci] * Cs, sc, A0.ravel()[ci], Lc, Uc,
                     W['fb'][kk.ravel()[ci]], wc, wnu)
            if ideal:
                exact = np.abs(W['fb'][kk.ravel()[ci]] - ftn.ravel()[ci][:, None]) < 1e-6
                val = np.where(exact | own4.ravel()[ci][:, None], val, 0)
            contrib_idx.append(ci); contrib_val.append(val)
        # straddling items: per tap
        si_ = np.nonzero(strad.ravel())[0]
        if len(si_):
            js, ms = jj.ravel()[si_], mm.ravel()[si_]
            dl = W['dly'][js, ms]; gg = W['gq'][js, ms]                      # (q,NTAP)
            Lq = np.maximum(L0.ravel()[si_][:, None] + dl, 0)
            Uq = np.minimum(H0.ravel()[si_][:, None] + dl, LW)
            Aq = A0.ravel()[si_][:, None] + dl
            ok = Uq > Lq
            rep = np.repeat(np.arange(len(si_)), NTAP)[ok.ravel()]
            vv = kernel_real((cf.ravel()[si_][:, None] * gg).ravel()[ok.ravel()],
                        np.repeat(ss.ravel()[si_], NTAP)[ok.ravel()],
                        Aq.ravel()[ok.ravel()], Lq.ravel()[ok.ravel()], Uq.ravel()[ok.ravel()],
                        W['fb'][np.repeat(kk.ravel()[si_], NTAP)[ok.ravel()]], wc, wnu)
            acc = np.zeros((len(si_), NB), complex)
            for bq in range(NB):
                acc[:, bq] = (np.bincount(rep, vv[:, bq].real, len(si_))
                              + 1j * np.bincount(rep, vv[:, bq].imag, len(si_)))
            contrib_idx.append(si_); contrib_val.append(acc)
        for ii, vv in zip(contrib_idx, contrib_val):
            tgt = (widx.ravel()[ii][:, None] * NB + np.arange(NB)[None, :]).ravel()
            Yr += np.bincount(tgt, vv.real.ravel(), tot)
            Yi += np.bincount(tgt, vv.imag.ravel(), tot)
            if store_int:
                io = ~own4.ravel()[ii]
                ints.append((a * 2 * M + pidx.ravel()[ii][io], vv[io]))
    Y = (Yr + 1j * Yi).reshape(N * P, 2, M, NB)
    if not store_int:
        return Y, None
    # assemble interference per (pair, digit, rx)
    npairs = len(t_idx)
    Ip = np.zeros((npairs * 2 * M, NB), np.complex64)
    for idx, vv in ints:
        np.add.at(Ip, idx, vv.astype(np.complex64))
    return Y, Ip.reshape(npairs, 2, M, NB)


# ---------------------------------------------------------------- detection
def noise_for(W, cfg):
    N, M = W['N'], W['M']
    Lc = noise_chol(cfg['window'])
    out = np.empty((N, P, 2, M, NB), complex)
    for k in range(N):
        g = np.random.default_rng([W['seed'], 8, k])
        wn = (g.standard_normal((P, 2, 8, NB)) + 1j * g.standard_normal((P, 2, 8, NB))) / np.sqrt(2)
        out[k] = (wn[:, :, :M] @ Lc.T)
    return out


def detect(W, cfg, Y, noise):
    """Y: (N*P,2,M,NB) normalised noise-free.  returns symbols (N,P) (-1 erasure), hard digit errors."""
    N, M = W['N'], W['M']
    Yn = Y.reshape(N, P, 2, M, NB) + noise
    zc = (np.conj(W['hc'])[:, None, None, :, :] * Yn).sum(3)              # (N,P,2,NB)
    met = np.abs(zc) ** 2                                                  # noise-only ~ Exp(1)
    eta = np.full(N, -np.log(cfg['pfa_bin']))
    b = met.argmax(-1)
    mx = np.take_along_axis(met, b[..., None], -1)[..., 0]
    dig = np.take_along_axis(W['dig_of_bin'], b.reshape(N, -1), 1).reshape(N, P, 2)
    era = mx < eta[:, None, None]
    sym = 15 * dig[..., 0] + dig[..., 1]
    bad = era.any(-1) | (sym > P - 1)
    sym = np.where(bad, -1, sym)
    true_d = np.stack([W['sym'] // 15, W['sym'] % 15], -1)
    return sym, (dig != true_d)


def lagrange_best(xs, ys):
    n = len(xs); C = COMBS[n]; X = xs[C]; Yc = ys[C]
    num_all = (xs[None, None, :] - X[:, :, None]) % P
    val = np.zeros((len(C), n), np.int64)
    for i in range(4):
        ot = [q for q in range(4) if q != i]
        num = num_all[:, ot[0]] * num_all[:, ot[1]] % P * num_all[:, ot[2]] % P
        den = ((X[:, i] - X[:, ot[0]]) * (X[:, i] - X[:, ot[1]]) % P * (X[:, i] - X[:, ot[2]])) % P
        val = (val + Yc[:, i:i + 1] * num % P * INV[den][:, None]) % P
    return (val == ys[None, :]).sum(1).max()


def decode_all(W, sym_raw, sym_cln, avail):
    """Errors-and-erasures decoding on the time grid of each phone's own pulses.
    Returns decode time (s, inf if none) and wrong flag."""
    N = W['N']
    truth = W['sym']; tev = W['S'] + TP
    Pval = truth
    tri = np.triu(np.ones((P, P), bool))          # tri[w,e]: window w available at eval e
    tdec = np.full(N, np.inf); wrong = np.zeros(N, bool)
    for k in range(N):
        use = avail[k][:, None] <= tev[k][None, :]
        eff = np.where(use, sym_cln[k][:, None], sym_raw[k][:, None])          # (w,e)
        valid = (eff >= 0) & tri
        n = valid.sum(0)
        A = ((eff == truth[k][:, None]) & valid).sum(0)
        e = n - A
        ok = (A >= 6) & (A - e >= 4)
        e_ok = np.argmax(ok) if ok.any() else P
        e_wr = P
        # structured: other phones' cubics
        mr = ((sym_raw[k][None, :] == Pval) | (sym_cln[k][None, :] == Pval)).sum(1)
        mr[k] = 0
        same = (W['coef'] == W['coef'][k]).all(1); mr[same] = 0
        for jc in np.nonzero(mr >= 6)[0]:
            AQ = ((eff == Pval[jc][:, None]) & valid).sum(0)
            okq = (AQ >= 6) & (2 * AQ - n >= 4)
            if okq.any():
                e_wr = min(e_wr, np.argmax(okq))
        # exhaustive while few symbols
        last = None
        for ev in range(min(e_ok, e_wr)):
            if n[ev] < 6 or n[ev] > 14 or e[ev] < 3:
                continue
            idx = np.nonzero(valid[:, ev])[0]
            key = (tuple(idx), tuple(eff[idx, ev]))
            if key == last:
                continue
            last = key
            best = lagrange_best(idx.astype(np.int64), eff[idx, ev].astype(np.int64))
            if best >= 6 and 2 * best - n[ev] >= 4:
                e_wr = ev; break
        if e_wr < e_ok:
            tdec[k] = tev[k, e_wr]; wrong[k] = True
        elif e_ok < P:
            tdec[k] = tev[k, e_ok]
    return tdec, wrong


# ---------------------------------------------------------------- one run
def run_one(args):
    name, seed = args
    cfg = dict(BASE); cfg.update(CONFIGS[name])
    t0 = time.time()
    W = make_world(cfg, seed)
    ti, si, own = build_pairs(W, cfg)
    Y, Ip = compute_Y(W, cfg, ti, si, own, store_int=cfg['sic'])
    noise = noise_for(W, cfg)
    sym, derr = detect(W, cfg, Y, noise)
    avail = np.full((W['N'], P), np.inf)
    tdec, wrong = decode_all(W, sym, sym, avail)
    if cfg['sic']:
        ip_pairs = np.nonzero(~own)[0]
        tgt = ti[ip_pairs]; src = si[ip_pairs]; jsrc = src // P
        eps = 10 ** (cfg['sic_resid_db'] / 20)
        for it in range(4):
            dec_ok = np.isfinite(tdec) & ~wrong
            sub = dec_ok[jsrc]
            fac = np.where(sub, eps * np.exp(1j * W['sic_ph'].ravel()[src]) - 1, 0)
            Yc = Y.copy()
            np.add.at(Yc, tgt, fac[:, None, None, None] * Ip[ip_pairs])
            av = np.zeros(W['N'] * P)
            np.maximum.at(av, tgt[sub], tdec[jsrc[sub]])
            sym_c, derr_c = detect(W, cfg, Yc, noise)
            tnew, wnew = decode_all(W, sym, sym_c, av.reshape(W['N'], P))
            done = np.array_equal(tnew, tdec) and np.array_equal(wnew, wrong)
            tdec, wrong = tnew, wnew
            if done:
                break
        derr = derr_c
    corr = (sym == W['sym']).sum(1); eras = (sym < 0).sum(1)
    res = dict(name=name, seed=seed, tdec=tdec, wrong=wrong, dep=W['dep'], x=W['x'], y=W['y'],
               F=W['F'], f0=W['f0'], Q=W['Q'], inc=W['inc'], gamma=W['gamma'],
               derr=derr.mean((1, 2)), corr=corr, eras=eras, npairs=len(ti), sec=time.time() - t0)
    return res


# ---------------------------------------------------------------- sanity checks
def pd_mfsk(M, es):
    return sum((-1) ** (k + 1) * comb(M - 1, k) / (k + 1) * exp(-k / (k + 1) * es) for k in range(1, M))


def sanity_lone(lines):
    """Lone phone at 3 m directly under the receivers, F=1 N, f0=170 Q=10, no echo/scatter."""
    cfg = dict(BASE); cfg.update(N=1, scatter_db=0.0, echo=(0.0, 0.0), kfac_db=300.0, mode='alone')
    W = make_world(cfg, 12345)
    W['x'][:] = 5; W['y'][:] = 5; W['dep'][:] = 3.0; W['F'][:] = 1.0; W['f0'][:] = 170.0; W['Q'][:] = 10.0
    G, l = path_gain(np.array([3.0]), 3.0)
    W['G'][:] = G[0]; W['tau'][:] = l[0] / CG
    W['dly'][:] = 0; W['gq'][:] = 0; W['gq'][..., 0] = 1
    gc = int(round(170 / GRID)); gidx = gc + np.arange(-7, 8)
    W['fb'][:] = GRID * gidx; W['dig_of_bin'][:] = gidx % 15
    tb = np.argsort(gidx % 15)
    W['fton'][0, :, 0] = W['fb'][0, tb[W['sym'][0] // 15]]; W['fton'][0, :, 1] = W['fb'][0, tb[W['sym'][0] % 15]]
    W['hc'][:] = 1 / np.sqrt(W['M'])
    out = {}
    for variant in ['flat -0.45 dB (README assumption)', 'true motor response']:
        if variant.startswith('flat'):
            Hsave = motor_H
            globals()['motor_H'] = lambda f, f0, Q: 10 ** (-0.45 / 20) * np.exp(-1j * np.arctan(2 * np.pi * Q / (np.pi * f0) * (f - f0)))
        ti, si, own = build_pairs(W, cfg)
        Y, _ = compute_Y(W, cfg, ti, si, own)
        if variant.startswith('flat'):
            globals()['motor_H'] = Hsave
        Yr = Y.reshape(P, 2, W['M'], NB)
        # signal energy in correct bin -> measured SNR
        tb_d = np.stack([tb[W['sym'][0] // 15], tb[W['sym'][0] % 15]], -1)
        es = (np.abs(np.take_along_axis(Yr, tb_d[:, :, None, None].repeat(W['M'], 2), 3)[..., 0]) ** 2).sum(-1)
        gam_db = 10 * np.log10(es.mean() * 4)
        Lc = noise_chol('rect'); g = np.random.default_rng(5)
        errs = 0; tot = 0
        truth = np.stack([W['sym'][0] // 15, W['sym'][0] % 15], -1)
        for rep in range(500):
            wn = (g.standard_normal((P, 2, W['M'], NB)) + 1j * g.standard_normal((P, 2, W['M'], NB))) / np.sqrt(2)
            met = np.abs((Yr + wn @ Lc.T).sum(2) / np.sqrt(W['M'])) ** 2
            dig = (gidx % 15)[met.argmax(-1)]
            errs += (dig != truth).sum(); tot += dig.size
        pe = errs / tot
        out[variant] = (gam_db, pe)
        lines.append(f"  {variant}: pulse SNR {gam_db:.2f} dB, digit error {pe:.4f} "
                     f"(+-{1.96*np.sqrt(pe*(1-pe)/tot):.4f}); analytic orthogonal 15-FSK at that SNR: "
                     f"{pd_mfsk(15, 10**(gam_db/10)/4):.4f}")
    return out


def sanity_collisions(lines, seeds=50):
    res = {}
    for ppm in [10.0, 20.0]:
        worst = 0; same_pairs = 0; npairs_gt1 = 0
        for sd in range(seeds):
            cfg = dict(BASE); cfg['drift_ppm'] = ppm
            W = make_world(cfg, sd)
            S = W['S']; N = W['N']
            flat = S.ravel(); own = np.repeat(np.arange(N), P)
            o = np.argsort(flat); fs = flat[o]; ow = own[o]
            cnt = {}
            for sh in range(1, 6):
                close = (fs[sh:] - fs[:-sh]) < 0.3
                a, b = ow[:-sh][close], ow[sh:][close]
                m = a != b
                for i, j in zip(np.minimum(a[m], b[m]), np.maximum(a[m], b[m])):
                    cnt[(i, j)] = cnt.get((i, j), 0) + 1
            key = W['u'] * P + W['v']
            for (i, j), c in cnt.items():
                if key[i] == key[j]:
                    same_pairs += 1; continue
                worst = max(worst, c); npairs_gt1 += c > 1
        res[ppm] = worst
        lines.append(f"  +-{ppm:.0f} ppm/phone ({2*ppm:.0f} ppm relative): max destructive collisions per pair "
                     f"per cycle = {worst}; pairs with >1: {npairs_gt1} over {seeds} seeds "
                     f"(identical (u,v) lines excluded: {same_pairs} pairs)")
    return res


def brute_check(lines, cfgname='c_full', seed=0, nwin=8, fs=20000.0, windows=('rect', 'hann')):
    """Compare closed-form bins against direct sampling of the real signal."""
    worst_all = 0
    for window in windows:
        cfg = dict(BASE); cfg.update(CONFIGS[cfgname]); cfg['window'] = window
        W = make_world(cfg, seed)
        ti, si, own = build_pairs(W, cfg)
        # choose target pulses: some with strong in-band interferers, some lone
        int_t = np.unique(ti[~own])
        inband = np.unique(ti[(~own) & (np.abs(W['f0'][si // P] - W['f0'][ti // P]) < 6)])
        g = np.random.default_rng(1)
        pick = (list(g.choice(inband, nwin - 4, replace=False)) + list(g.choice(int_t, 2, replace=False))
                + list(g.choice(np.setdiff1d(np.arange(W['N'] * P), int_t), 2, replace=False)))
        _, _, w2 = window_terms(window); sig_n = np.sqrt(NV ** 2 / 2 * w2)
        worst = 0; nchk = 0; winfo = None
        for tp in pick:
            m_ = np.isin(ti, [tp])
            Y, _ = compute_Y(W, cfg, ti[m_], si[m_], own[m_])
            Ya = Y[tp] * sig_n                                   # (2,M,NB) absolute
            k, r = tp // P, tp % P
            for d in range(2):
                t = (np.arange(int(LW * fs)) + 0.5) / fs
                ww = w_of_t(window, t)
                for m in range(W['M']):
                    x = np.zeros_like(t)
                    for sp in si[m_]:
                        j, rj = sp // P, sp % P
                        coef, s, anc, lo, hi, _, _ = pulse_pieces(W, np.array([j]), np.array([rj]))
                        off = (W['S'][j, rj] + W['tau'][j, m]) - (W['S'][k, r] + d * TD + WS + W['tau'][k, m])
                        for q in range(NTAP):
                            dl = W['dly'][j, m, q]; gg = W['gq'][j, m, q] * W['G'][j, m]
                            for pc in range(6):
                                a0 = anc[0, pc] + off + dl; l0 = lo[0, pc] + off + dl; h0 = hi[0, pc] + off + dl
                                msk = (t >= l0) & (t < h0)
                                if msk.any():
                                    x[msk] += (gg * coef[0, pc] * np.exp(s[0, pc] * (t[msk] - a0))).real
                    Yb = (ww * x)[None, :] * np.exp(-2j * np.pi * W['fb'][k][:, None] * t[None, :])
                    Yb = Yb.sum(1) / fs
                    ya = Ya[d, m]
                    lev = 20 * np.log10(np.abs(ya) + 1e-300)
                    sig = np.abs(ya) > 0.01 * sig_n          # every bin above -40 dB re noise
                    nchk += sig.sum()
                    err = np.abs(20 * np.log10(np.abs(Yb[sig])) - lev[sig])
                    if err.max() > worst:
                        worst = err.max(); winfo = (tp, d, m, 20*np.log10(np.abs(ya[sig][err.argmax()])/sig_n), lev.max()-20*np.log10(sig_n))
        lines.append(f"  {window}: {len(pick)} target pulses ({nwin-4} with in-band interferers) x 2 digits x "
                     f"{W['M']} rx; {nchk} bins checked (all above -40 dB re noise): "
                     f"max |closed form - sampled| = {worst:.4f} dB")
        worst_all = max(worst_all, worst)
    return worst_all


def prune_check(lines, seed=0):
    cfg = dict(BASE)
    out = []
    for pdb in [-30.0, -300.0]:
        cfg['prune_db'] = pdb
        W = make_world(cfg, seed)
        ti, si, own = build_pairs(W, cfg)
        Y, _ = compute_Y(W, cfg, ti, si, own)
        out.append((len(ti), Y))
    d = np.abs(out[0][1] - out[1][1]).max()
    lines.append(f"  pairs kept {out[0][0]} vs unpruned {out[1][0]}; max bin difference "
                 f"{d:.2e} noise-sigma (threshold -30 dB = 0.032)")
    return d


# ---------------------------------------------------------------- reporting
def summarize(results, seeds):
    names = [n for n in CONFIGS if n in results]
    ck = [h * 3600 for h in CHECK_H[:-1]] + [CYCLE + 60]
    S = {}
    for n in names:
        R = results[n]
        dec = np.array([[((r['tdec'] <= c) & ~r['wrong']).sum() for c in ck] for r in R])
        shal = np.array([[((r['tdec'] <= c) & ~r['wrong'] & (r['dep'] <= 3)).sum() for c in ck] for r in R])
        deep = np.array([[((r['tdec'] <= c) & ~r['wrong'] & (r['dep'] > 3)).sum() for c in ck] for r in R])
        nsh = np.array([(r['dep'] <= 3).sum() for r in R]); ndp = np.array([(r['dep'] > 3).sum() for r in R])
        wr = np.array([r['wrong'].sum() for r in R])
        S[n] = dict(dec=dec, shal=shal, deep=deep, nsh=nsh, ndp=ndp, wrong=wr,
                    sec=np.mean([r['sec'] for r in R]))
    return S


def fmt_ci(a):
    a = np.asarray(a, float)
    return f"{a.mean():.1f} [{np.percentile(a,5):.0f}-{np.percentile(a,95):.0f}]"


def write_outputs(results, S, lines, seeds):
    os.makedirs(OUT, exist_ok=True)
    # per-phone CSV
    keep = [n for n in ['a_alone', 'b_ideal', 'c_full', 'm_all'] if n in results]
    with open(os.path.join(OUT, 'per_phone.csv'), 'w', newline='') as fh:
        wtr = csv.writer(fh)
        wtr.writerow(['config', 'seed', 'phone', 'x_m', 'y_m', 'depth_m', 'F_N', 'f0_Hz', 'Q', 'clustered',
                      'gamma_alone_dB', 'digit_err', 'sym_correct', 'sym_erased', 'decode_h', 'wrong'])
        for n in keep:
            for r in results[n]:
                for i in range(len(r['dep'])):
                    wtr.writerow([n, r['seed'], i, f"{r['x'][i]:.2f}", f"{r['y'][i]:.2f}", f"{r['dep'][i]:.3f}",
                                  f"{r['F'][i]:.3f}", f"{r['f0'][i]:.2f}", f"{r['Q'][i]:.2f}", int(r['inc'][i]),
                                  f"{10*np.log10(r['gamma'][i]):.2f}", f"{r['derr'][i]:.4f}", r['corr'][i],
                                  r['eras'][i], '' if not np.isfinite(r['tdec'][i]) else f"{r['tdec'][i]/3600:.3f}",
                                  int(r['wrong'][i])])
    # summary CSV
    with open(os.path.join(OUT, 'summary.csv'), 'w', newline='') as fh:
        wtr = csv.writer(fh)
        hdr = ['config', 'label']
        for h in CHECK_H:
            hdr += [f'dec_{h}h_mean', f'dec_{h}h_p5', f'dec_{h}h_p95']
        hdr += ['shallow_68h_frac', 'deep_68h_frac', 'wrong_mean', 'wrong_max']
        wtr.writerow(hdr)
        for n, s in S.items():
            row = [n, LABEL[n]]
            for i in range(4):
                a = s['dec'][:, i]; row += [f"{a.mean():.2f}", f"{np.percentile(a,5):.0f}", f"{np.percentile(a,95):.0f}"]
            row += [f"{(s['shal'][:,3]/s['nsh']).mean():.4f}", f"{(s['deep'][:,3]/s['ndp']).mean():.4f}",
                    f"{s['wrong'].mean():.3f}", int(s['wrong'].max())]
            wtr.writerow(row)
    charts(results, S)
    with open(os.path.join(OUT, 'report.md'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')


def charts(results, S):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cols = {'a_alone': '#4a7ab8', 'b_ideal': '#58a55c', 'c_full': '#c44e52', 'm_all': '#8172b2',
            'm_all_hann': '#937860'}
    # 1. cumulative decoded vs time
    tg = np.linspace(0, 68, 273)
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=130)
    for n in ['a_alone', 'b_ideal', 'c_full', 'm_all']:
        if n not in results: continue
        curves = np.array([[((r['tdec'] <= t * 3600 + (60 if t == 68 else 0)) & ~r['wrong']).sum() for t in tg]
                           for r in results[n]])
        ax.fill_between(tg, np.percentile(curves, 5, 0), np.percentile(curves, 95, 0), color=cols[n], alpha=.18, lw=0)
        ax.plot(tg, curves.mean(0), color=cols[n], lw=2, label=LABEL[n])
    for h in CHECK_H: ax.axvline(h, color='0.8', lw=.8, zorder=0)
    ax.set_xlabel('hours since call'); ax.set_ylabel('phones decoded (of 200)')
    ax.set_title('KNOCKLINE v2: phones decoded vs time (mean, 5-95% over seeds)')
    ax.legend(frameon=False, loc='lower right'); ax.set_xlim(0, 68); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'decoded_vs_time.png')); plt.close(fig)
    # 2. success vs depth
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=130)
    edges = np.arange(1, 6.01, 0.5); mids = (edges[1:] + edges[:-1]) / 2
    for n in ['a_alone', 'b_ideal', 'c_full', 'm_all']:
        if n not in results: continue
        dep = np.concatenate([r['dep'] for r in results[n]])
        ok = np.concatenate([np.isfinite(r['tdec']) & ~r['wrong'] for r in results[n]])
        fr = [ok[(dep >= a) & (dep < b)].mean() for a, b in zip(edges[:-1], edges[1:])]
        ax.plot(mids, fr, 'o-', color=cols[n], label=LABEL[n])
    ax.axvline(3, color='0.6', ls='--', lw=1)
    ax.set_xlabel('depth (m)'); ax.set_ylabel('fraction decoded within 68 h'); ax.set_ylim(0, 1.02)
    ax.set_title('Success by depth'); ax.legend(frameon=False); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'success_by_depth.png')); plt.close(fig)
    # 3. configs bar chart (68 h and 24 h)
    names = [n for n in CONFIGS if n in S and not n.endswith('_b')]
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=130)
    yy = np.arange(len(names))[::-1]
    for i, (ci, c, lab) in enumerate([(3, '#4a7ab8', '68 h'), (2, '#9fbbe0', '24 h')]):
        m = [S[n]['dec'][:, ci].mean() for n in names]
        lo = [m_ - np.percentile(S[n]['dec'][:, ci], 5) for n, m_ in zip(names, m)]
        hi = [np.percentile(S[n]['dec'][:, ci], 95) - m_ for n, m_ in zip(names, m)]
        ax.barh(yy + (0.2 if i == 0 else -0.2), m, height=0.38, color=c, xerr=[lo, hi],
                error_kw=dict(lw=.8, capsize=2), label=lab)
    ax.set_yticks(yy); ax.set_yticklabels([LABEL[n] for n in names], fontsize=8)
    ax.set_xlabel('phones decoded (of 200), mean with 5-95% range'); ax.legend(frameon=False)
    ax.grid(axis='x', alpha=.2); ax.set_title('Runs, mitigations and sweeps')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'configs.png')); plt.close(fig)
    # 4. near-far scatter for c_full
    if 'c_full' in results and 'a_alone' in results:
        A = results['a_alone']; C = results['c_full']
        dep = np.concatenate([r['dep'] for r in A]); gam = np.concatenate([10 * np.log10(r['gamma']) for r in A])
        okA = np.concatenate([np.isfinite(r['tdec']) & ~r['wrong'] for r in A])
        okC = np.concatenate([np.isfinite(r['tdec']) & ~r['wrong'] for r in C])
        fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=130)
        ax.scatter(dep[okA & okC], gam[okA & okC], s=3, c='#9fbbe0', label='decoded in (a) and (c)')
        ax.scatter(dep[~okA], gam[~okA], s=3, c='0.75', label='fails even alone')
        ax.scatter(dep[okA & ~okC], gam[okA & ~okC], s=9, c='#c44e52', label='lost to collisions/leakage')
        ax.set_xlabel('depth (m)'); ax.set_ylabel('pulse SNR alone (dB)')
        ax.set_title('Which phones are lost (c vs a), all seeds'); ax.legend(frameon=False, markerscale=3)
        ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(os.path.join(OUT, 'lost_phones.png')); plt.close(fig)


def report(S, lines, seeds):
    L = lines
    L.append(f"\n## Phones decoded (of 200): mean [5th-95th percentile over {seeds} seeds]\n")
    L.append("| run | 6 h | 12 h | 24 h | 68 h | <=3 m @68h | >3 m @68h | wrong decodes (mean/max) | s/run |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for n, s in S.items():
        cells = [fmt_ci(s['dec'][:, i]) for i in range(4)]
        sh = f"{100*(s['shal'][:,3]/s['nsh']).mean():.1f}%"
        dp = f"{100*(s['deep'][:,3]/s['ndp']).mean():.1f}%"
        L.append(f"| {LABEL[n]} | " + " | ".join(cells) + f" | {sh} | {dp} | {s['wrong'].mean():.2f}/{s['wrong'].max()} | {s['sec']:.0f} |")
    def diff(a, b, tag):
        if a in S and b in S:
            d = S[a]['dec'] - S[b]['dec']
            se = d.std(0, ddof=1) / np.sqrt(len(d))
            cells = [f"{d[:,i].mean():.1f} +-{1.96*se[i]:.1f}" for i in range(4)]
            L.append(f"| {tag} | " + " | ".join(cells) + " |")
    L.append("\n## Paired differences (phones; mean +-95% CI of the mean)\n")
    L.append("| comparison | 6 h | 12 h | 24 h | 68 h |"); L.append("|---|---|---|---|---|")
    diff('a_alone', 'b_ideal', 'collision loss (a - b)')
    diff('b_ideal', 'c_full', 'LEAKAGE loss (b - c)')
    diff('a_alone', 'c_full', 'total loss (a - c)')
    for m in ['m_hann', 'm_pc', 'm_sic', 'm_8rx', 'm_all', 'm_all_hann']:
        diff(m, 'c_full', f'gain of {LABEL[m]}')
    diff('s_clust0_b', 's_clust0', 'leakage loss, clustering 0%')
    diff('s_clust90_b', 's_clust90', 'leakage loss, clustering 90%')
    diff('s_drift20_b', 's_drift20', 'leakage loss, drift +-20 ppm')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=50)
    ap.add_argument('--procs', type=int, default=os.cpu_count())
    ap.add_argument('--only', default='')
    ap.add_argument('--skip-checks', action='store_true')
    a = ap.parse_args()
    names = [n for n in CONFIGS if not a.only or n in a.only.split(',')]
    lines = ["# KNOCKLINE v2 near-far / leakage simulation -- results", "",
             f"seeds: {a.seeds}; configs: {len(names)}; generated by v2_leak.py", ""]
    if not a.skip_checks:
        lines.append("## Sanity checks\n\nLone phone, 3 m, F = 1 N, f0 = 170 Hz, Q = 10, 4 receivers, no echo/scatter:")
        sanity_lone(lines)
        lines.append("\nSchedule collisions (pulse starts within 0.3 s), 200 phones, call alignment 0.15 s:")
        sanity_collisions(lines, a.seeds)
        lines.append("\nClosed-form leakage vs brute-force sampling (real signal, 20 kHz):")
        brute_check(lines)
        lines.append("\nInterferer pruning (seed 0, c_full):")
        prune_check(lines)
        print('\n'.join(lines), flush=True)
    tasks = [(n, s) for n in names for s in range(a.seeds)]
    results = {n: [] for n in names}
    t0 = time.time()
    with Pool(a.procs) as pool:
        for i, r in enumerate(pool.imap_unordered(run_one, tasks, chunksize=1)):
            results[r['name']].append(r)
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(tasks)} runs, {time.time()-t0:.0f} s", flush=True)
    for n in names:
        results[n].sort(key=lambda r: r['seed'])
    S = summarize(results, a.seeds)
    report(S, lines, a.seeds)
    write_outputs(results, S, lines, a.seeds)
    print('\n'.join(lines[-60:]))


if __name__ == '__main__':
    main()
