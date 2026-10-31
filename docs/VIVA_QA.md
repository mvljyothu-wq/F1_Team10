# Likely viva questions, with answers

Answers are short on purpose: say them in your own words, then point at the code.
Numbers from the experiments are in `results/RESULTS.md`; quote those, not memory.

---

## A. UDP and the reliability layer

**1. Why build reliability on UDP instead of just using TCP?**
The goal is to study and control the reliability mechanisms themselves - in
particular the retransmission timeout. TCP hides them inside the kernel. On UDP
we decide sequence numbers, ACKs, windows and timers in our own code, so we can
swap the timeout estimator and measure the effect. (Real protocols do this too:
QUIC is reliable transport built on UDP.)

**2. What does UDP give you and what does it not?**
It gives ports (multiplexing), a datagram boundary and an optional 16-bit
checksum. It does not give delivery guarantees, ordering, duplicate suppression,
flow control or congestion control.

**3. Why Selective Repeat rather than Stop-and-Wait or Go-Back-N?**
Stop-and-Wait has one packet in flight, so throughput is at most payload/RTT
(1200 B / 300 ms = 4 KB/s on the high-latency profile). Go-Back-N resends the
whole window after one loss and discards out-of-order packets, wasting bandwidth
under loss and reordering. Selective Repeat keeps the pipe full, buffers
out-of-order data, and resends only the missing packet - and each retransmission
is caused by exactly one timer decision, which is what we want to study.

**4. How do you handle lost ACKs?**
Every ACK carries a cumulative field (lowest chunk not yet received). A later ACK
therefore confirms everything before it even if earlier ACKs were lost. If the
loss was the last ACK of a chunk, the sender's timer fires, it retransmits, and
the receiver - which already has the chunk - just ACKs again.

**5. How are duplicates and out-of-order packets handled?**
`ReceiveBuffer.offer`: below `expected` or already buffered = duplicate (ACKed,
not written); inside the window = buffered until the gap is filled; beyond the
window = dropped. In-order data is written immediately and the buffer releases
any consecutive chunks that follow.

**6. Why 32-bit sequence numbers without wrap-around?**
Sequence numbers count chunks. 2^32 chunks x 1200 B is about 5 TB, so wrap-around
cannot happen for realistic files; the sender refuses larger files. This avoids
modular-arithmetic bugs. Selective Repeat's rule "window <= half the sequence
space" is trivially satisfied (window <= 4096).

**7. Why a 1200-byte payload?**
24-byte header + 1200 + optional 16-byte HMAC + 8 (UDP) + 20 (IPv4) = 1268 bytes,
below 1280, the minimum MTU of every IPv6 link, and well below Ethernet's 1500.
So datagrams are never fragmented by IP. Fragmentation is bad because losing one
fragment loses the whole datagram. Overhead is about 2%.

**8. How does the transfer end, and what if the FIN_ACK is lost?**
After all chunks are ACKed the sender sends FIN. The receiver checks all chunks
arrived and the SHA-256 matches, renames the file atomically and replies
FIN_ACK(OK). The receiver keeps the finished transfer for a linger period and
re-sends the same FIN_ACK if a duplicate FIN arrives (like TCP's TIME_WAIT).

**9. Can the sender ever report success when the file is wrong?**
No. Success is only reported when a FIN_ACK with status OK arrives, and the
receiver only sends that after verifying the SHA-256 of what it wrote. A hash
mismatch gives HASH_MISMATCH, missing chunks give INCOMPLETE, silence gives an
error after retries. Tests cover each case. The experiment harness additionally
re-hashes every received file independently.

**10. What happens if the sender or the receiver crashes?**
Sender crash: the receiver's idle timeout (30 s) deletes the `.part` file.
Receiver crash: the sender's packets exceed `max_retries` and it exits with an
error (`test_receiver_interrupted_mid_transfer`). There is no resume.

---

## B. Integrity and security

**11. UDP checksum vs your CRC-32 vs SHA-256?**
UDP checksum: kernel, 16 bits, weak, optional on IPv4, does not see corruption
introduced above UDP (e.g. by our proxy). CRC-32: our per-packet check over
header + payload; detects accidental corruption (all single-bit errors and
bursts up to 32 bits). SHA-256: over the whole file; proves the reassembled file
is bit-identical. None of them authenticates the sender.

**12. Why doesn't SHA-256 authenticate the sender?**
An attacker who can send packets can also send the hash of their own file in the
SYN. A hash proves integrity relative to the claimed hash, not who claimed it.
Authentication needs a secret: our optional HMAC-SHA256 with a pre-shared key.

**13. How does the HMAC work and what are its limits?**
Each packet carries a 16-byte truncated HMAC-SHA256 of header + payload, keyed
with a pre-shared key read from a file. Without the key you cannot produce a
valid tag (`test_forged_packet_with_recomputed_crc_rejected`). Limits: key
distribution is manual; there is no encryption (contents are readable); replay
protection is partial - packets are bound to a random transfer ID and the
receiver refuses IDs it finished recently, but a full defence needs nonces or
timestamps.

**14. How do you stop a malicious datagram from hurting the receiver?**
`decode` validates length, magic, version, flags, length consistency and CRC
before anything else, and only raises `PacketError`. `recvfrom` reads at most
1241 bytes. The reorder buffer holds at most `window` chunks. SYN limits file
size, window and number of concurrent transfers. Filenames are rejected if they
contain path separators, `..` or control characters, and files are created with
mode `x` inside the output directory. Packets for a transfer must come from the
address that started it.

**15. Why is the model stored as JSON and not pickle/joblib?**
Unpickling can execute arbitrary code, so a tampered model file would give
code execution. JSON contains only numbers and lists; the loader also checks
every index so tree traversal cannot loop or go out of range.

---

## C. RTT, RTO and backoff

**16. Define RTT and RTO.**
RTT: time from sending a packet to receiving its ACK. RTO: how long the sender
waits for that ACK before retransmitting.

**17. Write the RFC 6298 equations.**
First sample R: SRTT = R, RTTVAR = R/2. Then RTTVAR = 3/4 RTTVAR + 1/4 |SRTT - R|
(using the old SRTT), SRTT = 7/8 SRTT + 1/8 R, RTO = SRTT + max(G, 4 RTTVAR),
clamped to [RTO_min, RTO_max]. Before any sample RTO = 1 s.

**18. Why K = 4?**
RTTVAR is a smoothed mean deviation. Four deviations above the mean is
conservative enough that ordinary jitter rarely causes a timeout.

**19. What is Karn's algorithm and where is it in your code?**
If a packet was retransmitted, its ACK might belong to either copy, so its RTT
is ambiguous and is not used. In `SendWindow.on_ack`: a sample is returned only
if `entry.retx == 0`. The same rule applies to SYN/FIN in `_exchange`.

**20. Why exponential backoff, and how do per-packet timers complicate it?**
Repeated timeouts suggest the path is much worse than the estimate; doubling
backs off fast. With 64 timers, one outage would double the RTO 64 times. The
manager keeps an epoch number: only a timeout of a packet sent in the current
epoch raises the level, so the RTO doubles once per round. A valid RTT sample
resets it and re-arms pending timers.

**21. Why is your RTO_min 30 ms and not 1 s as in the RFC?**
The 1 s floor suits the Internet and delayed ACKs. Our simulated RTTs are 20-300
ms; with a 1 s floor both estimators would always output 1 s and the comparison
would be meaningless. It is a documented deviation.

**22. What is a spurious retransmission?**
Resending a packet that was not lost - its ACK (or the packet itself) was only
delayed, or its ACK was lost. The proxy counts these exactly as "a retransmission
of a chunk whose earlier copy already reached the receiver intact".

---

## D. The machine-learning estimator

**23. What exactly does the model predict?**
At each decision point (when an RTT sample arrives at time t) it predicts the
95th percentile of the RTTs of packets sent in the next 0.5 s. A timeout equal to
that would let about 95% of delivered packets be ACKed before their timer fires.

**24. Why a high quantile and not the mean RTT?**
The RTO is a threshold. Using the mean would make about half of all packets time
out. The 95th percentile states an explicit budget for spurious timeouts.

**25. Isn't using future RTTs leakage?**
The future is used only to build the training label, offline. Features come
only from the past (they are computed by the sender at time t and logged). At
runtime the model never sees anything from the future.

**26. What are the features and why those?**
log SRTT, RTTVAR/SRTT, last/mean/std/p90/max/min of the last 16 RTTs divided by
SRTT, the RTT trend (slope), the retransmission ratio of the last 64 sends, time
since the last timeout, and how many samples we have. All are observable at the
sender. They are mostly relative to SRTT so the model learns shapes, not speeds.
Not used: the simulator's loss rate, the profile name, one-way delays - the
sender cannot know these, so using them would be leakage.

**27. How did you avoid leakage between training and test?**
By splitting whole runs by seed before training (train 0-19, validation
100-105, test 200-205, experiments 1000+). Adjacent rows in one run are nearly
identical; a random row split would put near-copies in both sets and inflate
accuracy.

**28. Why Random Forest?**
Handles non-linear interactions, needs little tuning, gives feature importances,
and is easy to export and evaluate without scikit-learn. We compared two sizes
and two target formulations on validation runs only.

**29. Direct prediction or correction to the conventional RTO?**
Both were trained: direct = log(target); correction = log(target / RTO_conv).
The one with lower validation error (after margin calibration and the safety
policy) was kept - see `models/rto_model.json -> metadata.candidates`. The
correction form is scale-free and keeps the formula as the anchor; tree models
cannot extrapolate beyond the targets they saw.

**30. How is the safety margin chosen?**
On validation runs: the smallest margin (1.00, 1.05, ...) whose spurious-timeout
fraction after the safety policy is at most max(conventional's fraction, 1%).

**31. What stops the model from doing something dangerous?**
`apply_safety`: never below rto_min, 0.5 x RTO_conv, or the largest RTT seen in
the last 16 samples; never above 2 x RTO_conv or rto_max. Backoff is applied on
top, identically in both modes. Any failure (no model, warm-up, exception,
NaN/inf) falls back to the conventional value and is counted.

**32. Online vs offline?**
Training is offline (`train_model.py`). At runtime the model is frozen and only
predicts from recent history, at most every 10 ms; it does not learn during a
transfer.

**33. Lower MAE means a better protocol?**
Not necessarily. Errors are asymmetric: too short causes spurious retransmissions
and backoff, too long only delays loss recovery. The protocol's behaviour also
depends on the sequence of decisions. That is why we report both offline error
and end-to-end transfer metrics.

---

## E. Experiments and results

**34. How do you make the comparison fair?**
Same file, same seed, same profile for both modes. The proxy's decision for a
packet copy is a hash of (seed, direction, type, seq, copy number), so "the 2nd
copy of chunk 117 is dropped" in both runs. The mode order alternates by seed.
Evaluation seeds are disjoint from training seeds.

**35. Why paired tests and a Wilcoxon test?**
Each seed gives a pair (conventional, ML) under identical conditions, so the
difference within a pair removes most noise. With 10 pairs we cannot assume
normality; the exact Wilcoxon signed-rank test makes no such assumption.

**36. What did you find?**
Quote `results/RESULTS.md` and `docs/RESULTS_DISCUSSION.md`. State where ML
helped, where it made no difference, where it was worse, and the mechanism.

**37. What are the main limitations?**
Single machine and loopback; user-space delay simulation (no queueing or
bandwidth limits); no congestion control; trained on our own simulator, so
real-network generalisation is unproven; 10 seeds per profile; Python timer
resolution.

**38. What would you do next?**
Test over real links / `tc netem`; add congestion control so RTO interacts with
sending rate; train on traces from real networks; try a quantile-regression loss
directly instead of label-then-regress; evaluate fast retransmit (duplicate-ACK
detection), which would reduce reliance on timeouts altogether.

---

## F. Version 2 (RACK, ML v2, live lab)

**39. What is RACK and why did you implement it?**
RACK (RFC 8985) detects loss by time instead of counting duplicate ACKs. A
packet is lost if a packet sent after it has already been delivered and it has
been outstanding longer than one RTT plus a reordering window (min RTT / 4,
adaptive). Linux TCP uses it. It repairs most losses without waiting for the
retransmission timer, which attacks the same problem as the ML estimator from
the protocol side.

**40. Why not the classic "3 duplicate ACKs" fast retransmit?**
I implemented it first. Our simulator jitters every packet independently, so
packets sent back to back arrive heavily reordered, and "3 later packets
arrived" fired thousands of false resends per transfer. Time-based detection
tolerates reordering up to the reordering window and adapts that window when
it detects its own mistakes.

**41. How does RACK know a resend was unnecessary without DSACK?**
If the ACK for a resent packet arrives sooner than the smallest RTT ever
measured after the resend, it cannot be for the new copy, so the original
arrived. That counts as a spurious resend and widens the reordering window.

**42. What exactly changed from ML v1 to v2?**
Three things, each answering a measured v1 failure. (1) Quantile loss, so
predicting too short costs q/(1-q) times more than predicting too long.
(2) A second data-collection round with v1 in control, so the model sees the
situations an ML estimator creates (covariate shift). (3) The margin and quantile
level chosen on real validation transfers with a rule fixed in advance (waste
no more than 1.15x RFC, then best goodput), because v1's offline metrics did not
predict online waste. Features, label and safety policy stay the same, so the
comparison isolates these changes.

**43. Is training on v1-controlled data cheating?**
No. It uses the same training seeds (0-19) and validation seeds (100-105).
Evaluation seeds (200+ offline, 1000+ end to end) are never touched.

**44. What does the live lab prove?**
That the results are not canned: anyone can start a real transfer from the
browser and watch every RTT sample, timeout and RACK resend arrive. It uses
exactly the experiment code path (receiver and proxy processes, real sender).
The published page, which cannot reach a local server, plays back recorded
experiment runs instead and says so.
