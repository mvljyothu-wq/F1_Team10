# Report structure, presentation outline and demo script

## 1. Project report structure

1. **Title page and abstract** (150-200 words): problem, approach (reliable
   transport over UDP; RFC 6298 vs ML RTO), evaluation method (paired trials
   on 7 simulated profiles), headline finding (copy from RESULTS_DISCUSSION.md),
   one sentence of limitations.
2. **Introduction**
   - Why reliability is needed on UDP; why the RTO matters (spurious
     retransmissions vs slow recovery).
   - Objectives (brief section 2) and contributions.
3. **Background**
   - UDP vs TCP; ARQ protocols (stop-and-wait, Go-Back-N, Selective Repeat).
   - RTT estimation: Jacobson/Karels, RFC 6298, Karn's algorithm, backoff.
   - Related ML work on RTO / congestion (cite 3-5 papers you actually read).
4. **System design** (from docs/DESIGN.md)
   - Architecture diagram (section 2), packet format table (section 5),
     workflow sequence diagram (section 4), failure handling table.
   - Design decisions table (decision log).
5. **Implementation**
   - Module overview (README structure); key algorithms with short code
     excerpts: `SendWindow.on_ack` (Karn), `TimeoutManager.on_timeout` (epoch
     backoff), `apply_safety`, `Impairment.decide` (hash-keyed fate).
   - The backoff bug and its fix (CODE_WALKTHROUGH section 4.3) - examiners like
     honest debugging stories.
6. **ML methodology**
   - Problem formulation, target construction, features, leakage control,
     seed-based splits, model selection, margin calibration, JSON export.
   - Offline results table and plots (`offline_*.png`, `feature_importance.png`).
7. **Experimental evaluation**
   - Setup (machine, Python version, profiles table, seeds, file size, trials).
   - Metric definitions (experiments/metrics.py docstring).
   - Results table (results/RESULTS.md), plots: goodput, retransmissions,
     duration, unnecessary retransmissions, recovery delay, change heatmap,
     RTT/RTO timelines.
   - Discussion: where ML helped / did not / hurt, and why.
8. **Testing** - test categories and counts; fuzzing; adverse-network tests.
9. **Security considerations** - validation, filenames, bounded resources,
   HMAC and its limits, no pickle; "not a production protocol".
10. **Limitations and future work** (VIVA_QA question 37-38).
11. **Conclusion**
12. **References**; **Appendix**: commands to reproduce, full metric tables.

## 2. Presentation outline (10-12 minutes, 12 slides)

1. **Title** - project name, your name, register number.
2. **Problem** - UDP drops/reorders/duplicates; the RTO dilemma in one picture
   (too short = spurious, too long = idle).
3. **Goal** - reliable transfer over UDP + "can ML choose a better RTO than
   TCP's formula?" - and an honest experiment to find out.
4. **Architecture** - the Mermaid diagram (sender, proxy, receiver, timeout manager).
5. **Protocol** - packet header table, Selective Repeat, cumulative + selective ACK.
6. **Reliability features** - CRC-32, SHA-256, atomic commit, failure handling;
   security (validation, HMAC, no pickle).
7. **Conventional RTO** - equations, Karn, backoff (mention the epoch fix).
8. **ML estimator** - target = P95 of next 0.5 s of RTTs; features; leakage
   control; safety policy and fallback.
9. **Fair comparison** - same seed, hash-keyed impairment, paired trials, Wilcoxon.
10. **Results** - `change_heatmap.png` + one bar chart; one sentence per finding.
11. **Live demo** (see below) or the `timeline_changing.png` plot if time is short.
12. **Limitations, future work, conclusion.**

## 3. Demonstration scenario (5 minutes)

Prepare beforehand: `source .venv/bin/activate`, terminal font large, run the
test suite once so files are cached.

1. **Tests pass** (30 s): `python3 -m unittest discover -s tests -t .`
   - "Every packet-format edge case, every single-bit flip, and end-to-end
     transfers under loss, duplication, reordering and corruption."
2. **Normal transfer** (30 s): `python3 demo.py --profile perfect`
   - Zero retransmissions, SHA-256 verified.
3. **Packet loss and retransmission** (1 min): `python3 demo.py --profile lossy`
   - Point at retransmissions and timeouts rising, RTO about 60 ms on a 50 ms
     RTT path, still SHA-256 verified.
4. **Adaptive timeout** (1.5 min): `python3 demo.py --profile changing --timeout-mode ml`
   - Watch the RTT jump from ~30 ms to ~240 ms and the conventional RTO and ML
     proposal follow it; "Selected" shows which one is in charge.
5. **Same network, both strategies** (1 min): `python3 demo.py --profile jitter_heavy --compare`
   - Explain that one seed is an anecdote, then show the results table.
6. **Robustness** (optional, 30 s): start `python3 receiver.py`, send with a
   wrong key: `python3 sender.py --file README.md --key-file wrong.key` -> clear
   failure, nothing written.
