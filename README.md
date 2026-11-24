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
| `v2_leak.py` | Near-far / spectral-leakage Monte Carlo of 200 phones over 68 h (results in `results/leak/`) |

Run any of them with `python3 <file>` (needs numpy; `v2_loc.py` also needs scipy).

## v2 design in brief

- **Schedule:** p = 211. Phone i pulses in slot (uᵢ + vᵢ·r) mod 211 in round r. Slot 5.5 s (5 s pulse + 0.5 s gap), so 211 rounds = 68 h.
- **Modulation:** each pulse is two 2.5 s digits. Each digit is one of 15 tones on a 0.4 Hz grid near the motor's resonance; digit = round(f/0.4) mod 15; symbol = 15·d₁ + d₂ in F₂₁₁.
- **Message:** a cubic polynomial over F₂₁₁ evaluated at r (4 symbols, 30.9 bits).
- **Collision guarantee:** if ρ̄p² + (ε₀+κ)/T_m < 1, any two phones destructively collide at most once per cycle. This holds for relative drift up to 20.6 ppm with no heartbeats.
- **Reach:** motor data about 2.9 m deep; coached rock strikes about 7 m; sledgehammer downlink about 10 m.
- **Energy:** 234.6 mWh of the 505 mWh budget over 72 h.

## Near-far leakage simulation (v2_leak.py): results

`python3 v2_leak.py --seeds 50 --procs 4` (about 3.5 h on 4 cores) writes `results/leak/report.md`, `summary.csv`, `per_phone.csv` and charts.

Model: closed-form bin integrals of every overlapping tone, including motor settling and ringdown, the negative-frequency image, and a 6-tap static echo channel (Rician direct tap, K = 3 dB). 4 receivers are combined coherently with learned gains. Decoding uses errors and erasures, needs ≥6 agreeing symbols, and searches for wrong decodes.

Sanity checks (all pass):
- **Lone phone at 3 m:** 14.4 dB SNR gives digit error 0.109; the analytic value is 0.107.
- **Collisions at ±10 ppm per phone (20 ppm relative):** at most 1 destructive collision per pair per cycle. At ±20 ppm, 735 pairs over 50 seeds have more than 1.
- **Closed form vs 20 kHz brute-force sampling:** the largest error is 0.0003 dB.

Phones decoded out of 200 (mean, with 5th–95th percentile over 50 seeds):

| run | 6 h | 12 h | 24 h | 68 h | ≤3 m | >3 m |
|---|---|---|---|---|---|---|
| (a) each phone alone | 48.8 [39–58] | 56.3 [44–65] | 61.6 [51–71] | 66.8 [56–78] | 76.5% | 4.9% |
| (b) collisions, ideal tones | 48.6 | 56.0 | 60.9 | 65.6 | 75.5% | 4.5% |
| **(c) full leakage** | **48.5 [39–58]** | **55.9 [43–65]** | **60.9 [49–70]** | **65.5 [56–74]** | **75.5%** | **4.4%** |
| (c) + 8 receivers | 71.0 | 78.6 | 83.8 | 89.0 [78–101] | 93.5% | 12.0% |
| (c) + power control + subtraction + 8 receivers | 70.7 | 78.6 | 84.2 | 89.8 [80–101] | 93.7% | 12.6% |

Findings:
- **Leakage is not the limit.** It costs 0.1 ± 0.2 phones at 68 h. Collisions cost 1.2 ± 0.3. Neither clustering (0–90%), echo spread (20–100 ms) nor ±20 ppm drift raises the leakage loss above 0.2 phones.
- **Depth is the limit.** About 76% of phones at ≤3 m decode, against about 4% deeper than 3 m. This agrees with the ~2.9 m motor reach, and slant paths make it worse.
- **8 receivers is the only mitigation that pays:** +23.5 phones.
- **Subtracting decoded phones** gives +1.3, which recovers the collision loss.
- **Power control** costs 0.7 phones: turning down near phones does not help when leakage is negligible.
- **The Hann window** costs 12.5 phones, because it gives up about 1.8 dB of SNR and there is no leakage for it to remove.
- **Scatter** helps deep phones by luck: 2 dB gives 62.1 phones, 6 dB gives 72.0.
- **Wrong decodes** average ≤0.08 per run, at most 1.

Assumptions: perfect acquisition and timing, frequency drift ignored, combining gains learned perfectly.
