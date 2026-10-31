# Results discussion

> **Version 2 update (section 11 below).** Three strategies were added on the
> same 70 network × seed pairs: ML v2, RFC + RACK and ML v2 + RACK (420
> transfers in total). Sections 1-10 describe the original three-strategy
> study and remain valid; section 11 covers what changed.

This interprets `results/RESULTS.md` (all numbers there are generated from
`results/trials.csv` and `results/offline_eval.*`). Figures are in
`results/plots/`. Numbers quoted here come from the run made on 24 Oct 2026
on the development machine; a rerun gives similar, not identical, values.

## 1. What was run

- **Offline:** 25,721 timeout decisions from 42 held-out test runs (seeds
  200-205, 7 profiles) that the model never saw in training or validation.
- **End-to-end:** 7 profiles x 10 seeds (1000-1009) x 3 timeout modes = 210
  transfers of a 1 MiB file: conventional RFC 6298 (K=4), the ML estimator, and
  an ablation arm - the RFC formula with K=2, i.e. a deliberately shorter timeout.
- **Integrity:** all 210 transfers succeeded, and every received file was
  SHA-256-verified by the receiver *and* re-hashed independently by the harness.
- **Same network for every mode:** measured data-loss rates per profile are
  essentially identical across modes (e.g. lossy 5.6% / 5.6% / 5.6%), which
  confirms the hash-keyed impairment did its job.

## 2. Headline

> On lossy and jittery paths the ML estimator delivered **+6% to +11% goodput**
> (lossy +5.9%, jitter_heavy +6.9%, dup_reorder +11.1%; each p = 0.002) by
> recovering from losses **6-21% faster**. It paid for this with **roughly twice
> as many unnecessary retransmissions** (+53% to +100% on those profiles, +356%
> on high_latency). On the rapidly **changing** path it gave **no goodput gain and
> 3.8x the unnecessary retransmissions**.
>
> The ablation shows most of the speed-up is simply "a shorter timeout": the
> K=2 formula gets similar or larger goodput gains. But the ML estimator reaches
> almost the same goodput with **far less waste under heavy-tailed jitter
> (-60% unnecessary retransmissions vs K=2, p = 0.002) and burst loss (-50%,
> p = 0.045)**. That is the one place where it is genuinely smarter, not just
> more aggressive.

`plots/tradeoff.png` shows this in one picture: up-left is better.

## 3. When the ML estimator helped, and why

**Heavy-tailed jitter and reordering (jitter_heavy, dup_reorder, also lossy).**
RFC 6298 adds 4 x RTTVAR. A single delay spike inflates RTTVAR for many samples
afterwards, so the conventional RTO stays well above what most packets need.
After a real loss, the sender waits that whole inflated timeout before
repairing it - and with Selective Repeat, a lost packet at the window base stalls
new data. The forest instead sees the *shape* of recent RTTs (p90, max, min
relative to SRTT) and proposes a timeout close to the actual upper tail.
Offline this shows as a much smaller error (jitter_heavy MAE 6.4 ms vs 12.2 ms;
dup_reorder 4.1 vs 12.8 ms); end-to-end as shorter loss-recovery delay
(jitter_heavy -16%, dup_reorder -21%) and higher goodput.

**Compared with K=2,** a uniformly shorter formula has to shrink the timeout
everywhere, including right after spikes, so it times out much more often
(jitter_heavy: 127 unnecessary retransmissions vs ML's 51). The ML estimator
shortens the timeout selectively. This is the evidence that it learned something
the formula's fixed K cannot express.

## 4. When it made no difference

- **stable:** both estimators sit at the 30 ms floor (`RTO_min`) on a ~22 ms path
  with no loss. No timeouts happen, so the estimator is irrelevant.
- **high_latency (goodput +0.4%):** throughput is limited by the window (64
  packets per 300 ms RTT), and recovering a loss is dominated by the RTT itself,
  so a slightly shorter RTO barely matters. Yet ML produced 4.6x as many
  unnecessary retransmissions here - offline it was already slightly *more*
  spurious than the formula on this profile (6.9% vs 5.5%). Here ML is a net
  loss: the same speed for more traffic.

## 5. When it hurt: the changing path

On `changing` (delay ramps from 15 to 120 ms and steps back, every 5 s) ML gave
-2.1% goodput (not significant) and 155 unnecessary retransmissions vs 41.
`plots/timeline_changing.png` shows why. During the high-delay phase, samples
arrive in bursts once per RTT; the ML timeout sits just above the most recent
burst of RTTs. While the delay is still ramping up, packets sent next take
longer than anything in that burst, and their timers fire just before their ACKs
arrive (the clusters of timeout ticks between 1.4 s and 1.9 s).

Two pieces of evidence that this is a generalisation problem, not bad luck:

- **Feature importances** (`plots/feature_importance.png`): `log_srtt`
  dominates (~0.79), `slope` is almost unused. The model mostly learned "the
  typical good timeout for this RTT level", not "the RTT is rising, wait longer".
- **Leave-one-profile-out:** a model trained without `changing` has a *higher*
  error than the plain formula on it (25.3 ms vs 18.9 ms). The model does not
  extrapolate to dynamics it has not seen, as expected from tree models.

## 6. Why offline accuracy did not predict online behaviour

Offline, the ML timeout was both more accurate (MAE 5.7 vs 10.0 ms) **and**
less spurious (5.1% vs 7.2% of future packets would exceed it). Online, it
caused *more* unnecessary retransmissions. This is the most important lesson of
the project, and the brief anticipated it ("lower prediction error does not
automatically imply a better transport protocol"). Contributing causes:

1. **Closed loop vs open loop.** Offline, each decision is scored independently.
   Online, delays are correlated: when a slow period starts, *every* in-flight
   packet's timer is about to fire, so one too-short timeout becomes dozens of
   retransmissions at once. A per-decision average does not weigh this.
2. **Karn and backoff feedback.** Every spurious timeout turns a packet into a
   retransmitted one, which then gives no RTT sample, so history becomes thinner
   exactly when conditions change.
3. **The conventional estimator is more conservative online than offline.**
   Offline we scored its *base* RTO; online, after a timeout its backed-off value
   persists until the next valid sample (visible as the long 500+ ms plateaus in
   the timeline), which suppresses further spurious timeouts.
4. **Covariate shift.** Training traces were recorded under the conventional
   estimator; under ML control the sender sees more retransmissions and fewer
   samples than any training run.
5. **Not every unnecessary retransmission is avoidable.** Lost ACKs cause them
   too (lossy has 2% ACK loss): the K=4 formula still has 16 per transfer there.

## 7. Cost

Sender CPU rose from 0.039 to 0.047 CPU-seconds per second of transfer (about
+20%, still under 5% of one core), because the forest (40 trees) is evaluated at
most every 10 ms. 5% of ML-mode decisions fell back to the conventional value.
`trials.csv` stores only the total, so the reasons were spot-checked with four
extra ML transfers (changing, jitter_heavy, high_latency, dup_reorder): every
fallback there was the 8-sample warm-up, with no model errors or invalid
predictions.

## 8. Is the ML estimator "better"?

It depends on what you value, and the honest answer is: **not in general.**

- If completion time is all that matters, on lossy/jittery paths ML is faster
  than RFC 6298 - but so is simply using K=2.
- If network efficiency matters (shared links, metered traffic), ML wastes more
  than RFC 6298 everywhere there is loss or jitter.
- Against an equally aggressive formula, ML is clearly better under heavy-tailed
  jitter and burst loss, similar on random loss and reordering, and worse on
  rapidly changing paths.
- Without the safety policy and fallback it would be risky to deploy: its worst
  behaviour came on conditions it was not trained for.

## 9. What would likely improve it (future work)

1. Calibrate the margin on **end-to-end** unnecessary retransmissions, not the
   offline spurious fraction (section 6).
2. Train on traces collected **under ML control** too (iterate: train, deploy,
   collect, retrain) to remove covariate shift.
3. Use an **asymmetric loss** (quantile/pinball loss on the RTO itself) so
   under-prediction costs more than over-prediction.
4. Add training data with **more varied dynamics** (ramps at different speeds),
   or features that emphasise trends.
5. Compare against **fast retransmit** (react to duplicate/selective ACKs), which
   would reduce the dependence on timeouts altogether.

## 10. Limitations

- One machine, loopback, user-space delay simulation: no queueing, no bandwidth
  limit, no cross traffic, about 1 ms timing accuracy. No congestion control.
- The model is trained and tested on the same simulator; real-network performance
  is unproven.
- 10 seeds per profile: enough for the large effects (p = 0.002 is the smallest
  possible with 10 pairs) but weak for differences of 1-2%.
- The K=2 arm was run after the first two arms, not interleaved with them.
- RTT samples come only from a packet's own ACK. When reverse-path reordering
  lets a later cumulative ACK cover a packet first, no sample is taken. The
  sender already knew that packet arrived, so its timer never mattered - but it
  means the logged RTT distribution slightly under-represents the slowest ACKs.

## 11. Version 2: RACK, ML v2 and their combination

Averages over the six lossy networks, relative to RFC 6298 (paired by seed;
waste = unnecessary resends):

| Strategy | Goodput | Waste |
|---|---|---|
| ML v1 | +4.6 % | 2.47× |
| RFC, K = 2 | +6.6 % | 3.30× |
| ML v2 | −2.2 % | 0.71× |
| RFC + RACK | +4.9 % | 1.30× |
| ML v2 + RACK | +4.3 % | 0.88× |

Per-network numbers and significance: `results/RESULTS.md`, section
"Paired comparisons".

**RACK is the dependable gain.** Detecting loss from the acknowledgement
pattern made RFC 6298 faster on every lossy network (+2.9 % to +7.2 %;
significant on five of six), with modest extra waste except under deliberate
reordering (dup_reorder +87 %), where some reordered packets still look lost.
It repairs losses without waiting for a timer, so it attacks the same problem
as the ML estimator from the protocol side, and more reliably.

**ML v2 does what its calibration rule asked, and no more.** Its rule was
fixed in advance: no more than 1.15× the formula's waste, then the best
goodput. On validation transfers that meant a margin of 1.15, which matched
the formula's speed with 43 % less waste. On the evaluation seeds it was
leaner still (0.71×) but slower on most networks (−2.2 % on average). It was
faster and leaner only under heavy jitter (+2.2 %, −20 % waste, both
significant). Offline it shows the same character: early-timeout share 0.5 %
against the formula's 7.2 %, at a larger average error (17.9 vs 10.0 ms),
because it deliberately waits longer.

**Together they are better than either.** With RACK repairing most losses
quickly, the timeout only handles what RACK cannot see (for example the last
packets of a window). There, v2's caution costs little and saves spurious
resends. ML v2 + RACK was never slower than the formula on any lossy network
(+0.2 % to +10.2 %), wasted less on five of six, and averaged +4.3 % goodput
with 12 % fewer unnecessary resends: the only strategy better on both counts.

**Cost on the fast path.** On the loss-free stable network, the sender is
CPU-bound in Python. ML v2 (300 trees evaluated up to every 10 ms) and RACK's
per-iteration scan of the window cost about 2 % goodput there (sender CPU per
second of transfer: RFC 0.039 s, ML v2 0.053 s, RACK 0.064 s, both 0.076 s).
A compiled implementation would remove most of this.

**Caveat.** The three new strategies were run after the original three, not
interleaved, so slow drift in machine load could shift results by a percent or
two between the two groups. Comparisons inside each group are interleaved.
