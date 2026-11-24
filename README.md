# KNOCKLINE

A protocol that lets trapped smartphones signal through earthquake rubble. Rescuers knock, and phones answer with timed vibration-motor pulses that carry the message in frequency.

## Files

| File | What it checks |
|---|---|
| `v1_model.py` | Rubble link budget: path gain and SNR vs depth |
| `v1_mac.py` | Collision bound (200 phones, p = 211), ghost lines, unsynchronized case |
| `v1_check.py` | Re-check of the deterministic (≥12 clean pulses) and unsynchronized bounds |
| `v1_ladder.py` | Field-size ladder (23 → 53 → 211) decode times |
| `v1_six.py` | Decode times and v1 72-hour energy budget |
| `v2_phy.py` | Dispersion check, motor bandwidth/Q, FSK error rates, frequency pilot |
| `v2_mac.py` | Drift-tolerant collision bound (20/40 ppm, with/without call alignment) |
| `v2_loc.py` | Monte Carlo of energy + dispersion-ruler + barometer localization |

Run any of them with `python3 <file>` (needs numpy; `v2_loc.py` also needs scipy).

## v2 design in brief

- **Schedule:** p = 211. Phone i pulses in slot (uᵢ + vᵢ·r) mod 211 in round r. Slot 5.5 s (5 s pulse + 0.5 s gap), so 211 rounds = 68 h.
- **Modulation:** each pulse is two 2.5 s digits. Each digit is one of 15 tones on a 0.4 Hz grid near the motor's resonance; digit = round(f/0.4) mod 15; symbol = 15·d₁ + d₂ in F₂₁₁.
- **Message:** a cubic polynomial over F₂₁₁ evaluated at r (4 symbols, 30.9 bits).
- **Collision guarantee:** if ρ̄p² + (ε₀+κ)/T_m < 1, any two phones destructively collide at most once per cycle. This holds for relative drift up to 20.6 ppm with no heartbeats.
- **Reach:** motor data about 2.9 m deep; coached rock strikes about 7 m; sledgehammer downlink about 10 m.
- **Energy:** 234.6 mWh of the 505 mWh budget over 72 h.

## Next task: near-far leakage simulation

Paste this into a new Claude Code session on this repo:

```
Read README.md. Plan, then write and run a
self-contained Python (numpy/scipy)
simulation of KNOCKLINE v2. Question: how
many of 200 phones decode their message
within 68 h under real spectral leakage
and near-far effects?

SETUP
- 200 phones, depth uniform 1-6 m, 20x20 m
  footprint; 4 surface receivers.
- Path gain G = sqrt(2/(pi*k*l))*exp(-a*l)
  *10^(-(d/1.5)*8/20); l = 1.3*slant dist,
  k=2.26/m, a=0.0566 Np/m; add 4 dB
  lognormal scatter.
- Source velocity = F*Y*G, Y=1.25e-6 m/(N s),
  F = 0.7-1.5 N. Receiver noise
  3.34e-8 m/s/rtHz.
- Motor: f0 from 4-6 model clusters +-2 Hz
  (60%) else uniform 150-240 Hz; Q 5-20;
  |H|^2 = 1/(1+Q^2(f/f0-f0/f)^2).
- Static echo channel, 20-100 ms spread.
- Clock drift +-10 ppm/phone (stress +-20);
  call alignment within 0.15 s.

SCHEDULE AND MODULATION: as in README.
- Receiver integrates 0.15-2.5 s of each
  digit, combines 4 receivers by learned
  gain.
- Errors-and-erasures decoding of the
  cubic, need >=6 agreeing symbols; count
  wrong decodes separately.

LEAKAGE
- Each bin = sum over overlapping tones of
  A*H*e^{j phi} * integral w(t)
  e^{j2pi(f_s-f_k)t} dt + noise (include
  motor settling, tau = Q/(pi*f0)).
- Check against brute-force sampling on a
  few slots (<=0.1 dB).

RUNS (50 seeds)
- (a) each phone alone (b) collisions,
  ideal tone separation (c) full leakage.
- Mitigations: tapered window; power
  control from downlink level; subtract
  decoded loud phones; 8 receivers.
- Sweeps: motor clustering, echo spread,
  scatter 2-6 dB, drift 20 vs 40 ppm.

SANITY CHECKS
- Lone phone at 3 m: digit error ~0.105
  at SNR 14.5 dB.
- <=1 destructive collision (starts within
  0.3 s) per pair per cycle at 20 ppm.

OUTPUT
- Phones decoded at 6/12/24/68 h with
  confidence range; success by depth
  (<=3 m and >3 m separately); loss caused
  by leakage; per-phone CSV; charts.
```
