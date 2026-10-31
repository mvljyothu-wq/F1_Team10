# Code walkthrough

A guided tour of the code, in the order a packet experiences it. Read this with
the files open. Every section names the file and function so you can point at
the exact lines during the viva.

---

## 0. The big picture in one paragraph

`sender.py` reads a file, cuts it into 1200-byte chunks and sends them as DATA
packets over UDP, keeping up to 64 in flight. `receiver.py` checks each packet,
puts chunks back in order, writes them to a temporary `.part` file and sends an
ACK for every packet. The sender keeps a timer per packet; if an ACK does not
arrive before the timer expires, it resends that packet. How long the timer
waits (the RTO) comes from `timeout/manager.py`, which uses either the classic
TCP formula or the ML model. At the end the receiver compares the SHA-256 of
what it wrote with the hash the sender announced, and only then renames the
file and reports success. `proxy.py` sits in the middle and pretends to be a bad
network.

---

## 1. Packets: `protocol/packet.py`

**What it does:** converts between Python objects (`Packet`) and bytes on the wire.

- `HEADER_FORMAT = "!HBBBBHIIII"` - the `!` means *network byte order* (big-endian).
  `H` = 2-byte unsigned, `B` = 1 byte, `I` = 4 bytes. That gives the 24-byte header:
  magic, version, type, flags, reserved, payload length, transfer id, seq, ack, checksum.
- `encode(pkt, key)` - validates every field (so we never send garbage), packs the
  header with checksum = 0, optionally appends an HMAC tag, computes CRC-32 over
  everything and writes it into bytes 20-23.
- `decode(data, key)` - the *security boundary*: every byte from the network
  passes through here. It checks, in order: minimum and maximum length, magic,
  version, flags, that `payload_len` matches the real length, the CRC, then the
  HMAC (if a key is configured), and finally the message type. Anything wrong
  raises `PacketError` - and *only* `PacketError`, which the fuzz test
  (`test_random_garbage_never_crashes`) checks with 5000 random datagrams.
- Why check the CRC before the type? So a random bit-flip is reported as
  "corrupted", not as "unknown type".
- `encode_syn` / `decode_syn` - the SYN carries file size, number of chunks,
  chunk size, window, SHA-256 and the filename. `decode_syn` re-checks that
  `total_chunks == ceil(file_size / chunk_size)`, so inconsistent metadata is
  rejected before any file is created.
- `peek(data)` - lets the proxy read type and sequence number without verifying
  anything (the proxy is "the network"; it has no key).

**Viva point:** CRC-32 detects accidental corruption but anyone can recompute it.
The test `test_forged_packet_with_recomputed_crc_rejected` shows an attacker
fixing the CRC after tampering - only the HMAC catches it.

---

## 2. The sliding window: `protocol/window.py`

Pure bookkeeping, no sockets and no clock, so it is easy to test.

### `SendWindow` (sender side)

- `base` = oldest unacknowledged chunk, `next_seq` = next never-sent chunk.
  The window rule is `next_seq < base + window`.
- `inflight` maps seq -> `InFlight` (first send time, last send time, timer
  deadline, number of retransmissions, backoff epoch).
- Timers live in a **heap** of `(deadline, seq)`. When a packet is ACKed or
  re-armed, its old heap entry is not removed (that would be slow); instead
  `_discard_stale_timers` skips entries whose deadline no longer matches. This
  is called *lazy deletion*.
- `on_ack(seq, cum_ack, now)`:
  1. the selective part - marks `seq` as acknowledged;
  2. the cumulative part - marks everything below `cum_ack` as acknowledged
     (this is what makes lost ACKs harmless);
  3. slides `base` forward over acknowledged chunks.

  It returns an RTT sample **only if the packet was never retransmitted** -
  this one `if entry.retx == 0` line *is* Karn's algorithm.
- `rearm_all(rto, epoch)` - restarts all timers after a backoff ends (see 4.3).

### `ReceiveBuffer` (receiver side)

- `expected` = the next chunk needed to continue writing in order.
- `offer(seq, data)` returns a status and the list of chunks now ready to write:
  in order -> write immediately (plus any buffered chunks that now follow);
  ahead but within the window -> buffer; already seen -> duplicate;
  beyond the window -> drop. At most `window` chunks are ever buffered, so a
  malicious sender cannot make the receiver use unbounded memory.

---

## 3. Sender and receiver: `protocol/reliability.py`

### The sender (`_Sender`)

`run()` does four things: open a socket and `connect()` it to the receiver (so
the kernel drops datagrams from anybody else), `_handshake()`, `_data_phase()`,
`_finish()`. Any network problem becomes a `TransferError`, which ends up in
`TransferResult.error` with `success=False` - there is no code path that
reports success without a FIN_ACK carrying status OK.

`_data_phase()` is the heart of the project. Each loop iteration:

1. **Retransmit expired packets.** `win.pop_expired(now)` returns packets whose
   timer fired. For each: tell the timeout manager (`tm.on_timeout`), give up if
   it has already been retransmitted `max_retries` times, otherwise resend it
   and re-arm its timer with the *current* RTO (which may now be doubled).
2. **Fill the window** with new packets while `win.can_send_new()`.
3. **Wait** on the socket, but only until the earliest timer deadline. This is
   why the program does not busy-wait: the operating system wakes us either
   when an ACK arrives or when the next timer is due.
4. **Process an ACK** with `win.on_ack`. If it produced a valid RTT sample,
   `_on_sample` feeds it to the timeout manager.

`_exchange()` is used for SYN and FIN: send, wait up to one RTO for the reply,
retry with backoff, give up after `handshake_retries`.

### The receiver (`Receiver`)

`serve()` loops on `recvfrom` with a 0.2 s timeout so it can also run
`housekeeping()` (drop silent transfers, forget finished ones after `linger`).

- `_on_syn` - validates *everything* before creating a file: transfer id not in
  use, not recently finished (replay protection), not too many transfers, valid
  SYN, safe filename (`safe_filename` rejects `/`, `\`, `..`, control
  characters), size and window limits. The file is opened with mode `"xb"`
  (fail if it exists), and the final name gets a ` (1)` suffix if needed.
- `_on_data` - checks the exact chunk length, offers it to the `ReceiveBuffer`,
  writes ready chunks, updates a running SHA-256, and ACKs. Duplicates are
  ACKed again (the previous ACK may have been lost); out-of-window packets are
  not.
- `_on_fin` - if all chunks arrived and the hash matches: `fsync`, then
  `os.replace` the `.part` file to its final name (atomic: the final file
  either exists completely or not at all). Otherwise delete the partial file.
  The FIN_ACK is cached so a duplicate FIN (because our FIN_ACK was lost) gets
  the same answer.

---

## 4. Timeouts: `timeout/`

### 4.1 `conventional.py` - RFC 6298

`update(rtt)` is five lines that implement the equations in DESIGN.md section 6.
Note that RTTVAR is updated **before** SRTT, using the old SRTT, exactly as the
RFC specifies (`test_hand_computed_values` checks this with numbers you can
verify on paper).

### 4.2 `features.py` - what the model sees

`History` keeps the last 16 RTT samples, the last 64 transmissions, and the time
of the last timeout. `compute_features` turns that into 12 numbers. Most are
divided by SRTT so the model learns *shapes* ("the last RTT was 1.4x the average")
rather than absolute speeds. This same function is used to log training data
and at runtime, so they cannot disagree.

### 4.3 `manager.py` - one place that answers "how long to wait?"

- `on_rtt_sample` - updates the conventional estimator (always, in both modes),
  stores the sample, resets backoff, and in ML mode asks the model for a new
  decision (at most every 10 ms, to keep CPU use low).
- `on_timeout(now, epoch_at_send)` - **the backoff rule.** With 64 packets in
  flight, one outage makes many timers fire together. If each doubled the RTO
  we would jump straight to 10 s. So the manager keeps an *epoch* number that
  changes whenever the backoff level changes; a timeout only raises the level if
  that packet was sent in the current epoch. Result: the RTO doubles once per
  round of timeouts, like TCP's single timer.

  *Bug story worth telling in the viva:* the first version compared levels
  instead of epochs. A packet retransmitted during an earlier backoff could push
  the level up again after it had been reset, and on the lossy profile the RTO
  climbed to 1.9 s on a 50 ms path - a 3 MB transfer took 22 s. Logging the RTO
  trace exposed it; switching to epochs and re-arming timers when backoff ends
  (`SendWindow.rearm_all`) brought loss recovery from 460 ms to about 60 ms and
  the transfer to 4.3 s. A regression test (`test_stale_packet_cannot_raise_level_after_reset`)
  now guards it.
- `current_rto()` = selected base RTO x 2^backoff_level, capped at `rto_max`.

### 4.4 `ml_estimator.py` - the model at runtime

- `ForestModel.load` reads JSON, not pickle. Pickle can execute code while
  loading, so a tampered model file would be a remote-code-execution bug.
  `_validate_tree` also checks that every child index is *after* its parent,
  which guarantees tree traversal terminates.
- `predict` walks each tree (`x <= threshold ? left : right`) and averages the
  leaves. Inputs are rounded to float32 first, because scikit-learn does that;
  `train_model.py` asserts the two give identical predictions.
- `MLTimeoutEstimator.decide` - every way the model could fail ends in a
  `Decision` with `source="fallback"` and the conventional value: no model,
  not enough history (warm-up), an exception, NaN/infinity. A good prediction
  goes through `apply_safety`.
- `apply_safety` - the bounds: never below `max(rto_min, 0.5 x RTO_conv, largest
  recent RTT)`, never above `min(rto_max, 2 x RTO_conv)`.

---

## 5. The simulated network: `simulation/`

- `profiles.py` - each profile is a list of phases (delay, jitter, loss,
  duplication, reordering, corruption, ACK delay) that repeats. `changing`
  ramps and steps the delay; `bursty_loss` adds bad periods with 50% loss.
- `impairment.py` - `decide()` hashes `(seed, direction, type, seq, copy number)`
  with BLAKE2b to get 8 random numbers. Because the fate of a packet copy does not
  depend on anything sent before it, two runs with different timeout strategies
  face the same network. (`test_order_independence` proves this.)
- `proxy_core.py` - a `select()` loop on two sockets plus a heap of delayed
  packets ordered by delivery time. It also records ground truth the sender
  cannot see: which retransmissions were unnecessary and how long each loss
  took to repair.

---

## 6. Machine learning pipeline: `training/`

1. `generate_dataset.py` - 224 real transfers (7 profiles x 32 seeds) with the
   conventional estimator, each logging every RTT sample *and* the 12 features
   at that moment. Seeds are split before training: 0-19 train, 100-105
   validation, 200-205 test.
2. `dataset.py` - builds the target for each logged decision at time *t*: the
   95th percentile of the RTTs of packets **sent** in the following 0.5 s. It
   uses `searchsorted` on the send times so the future is used only for the label.
3. `train_model.py` - fits Random Forests for two formulations (log-correction
   to RTO_conv, and direct log-RTO), chooses the safety margin on validation
   runs, picks the best candidate by validation error, exports it to JSON and
   checks the runtime model reproduces scikit-learn exactly.
4. `evaluate_model.py` - scores conventional, a trivial "recent max" baseline
   and the ML model on the untouched test runs.

---

## 7. Experiments: `experiments/`

- `harness.py` - `Lab` starts a receiver process; `run_trial` starts a proxy
  process with the given profile and seed, runs the sender, stops the proxy,
  reads its ground truth and re-hashes the received file independently.
- `run_experiments.py` - for each profile and seed, runs both modes (alternating
  which goes first) and appends one row per transfer to `results/trials.csv`.
- `metrics.py` - metric definitions, exact Wilcoxon signed-rank test, bootstrap
  confidence interval.
- `report.py` - plots and `results/RESULTS.md`, computed only from those files.

---

## 8. Version 2 additions

### 8.1 RACK loss detection: `protocol/window.py`

Turned on with `SenderConfig(fast_retransmit=True)` or `--fast-retransmit`.

- `_rack_update(entry, now)` runs for every packet newly acknowledged (selective
  or cumulative). It remembers the **send time of the most recently sent
  packet that has been delivered** (`rack_xmit`) and that packet's RTT
  (`rack_rtt`), plus the minimum RTT seen.
- `_rack_deadline(entry)` returns `None` for packets sent after `rack_xmit`
  (nothing newer has arrived, so there is no evidence yet). For older packets it
  returns `last_sent + rack_rtt + reo_wnd`.
- `detect_losses(now)` returns packets past their deadline. Each is reported
  once per transmission (`fast_done`).
- `next_rack_deadline()` is the **reorder timer**: the sender's loop wakes up at
  `min(next retransmission timer, next RACK deadline)`.
- Adaptation: if the ACK for a RACK-resent packet arrives sooner than the minimum
  RTT after the resend, the original copy must have arrived, so the resend was
  unnecessary. `reo_mult` grows, widening the reordering window
  (capped at one RTT); after 16 clean repairs it resets.

In `reliability.py`, step 4 of `_data_phase` resends RACK's losses with
`by_rack=True`. It does **not** call `on_timeout`, so there is no backoff.

*Story for the viva:* the first fast retransmit used TCP's classic
"3 duplicate ACKs" idea. Measured on the lossy network it sent about 3,400
resends for a 900-chunk file, almost all unnecessary. The reason is that our
proxy jitters each packet independently, which reorders bursts heavily
(Linux netem does the same). Time-based RACK fixed it.

### 8.2 ML v2: `training/train_model_v2.py`, `training/calibrate_v2.py`

1. `generate_dataset.py --mode ml --tag mlctl` repeats the training transfers
   with the **v1 model in control**, logging features at every sample
   (`log_features` now works in ML mode too).
2. `train_model_v2.py` trains `GradientBoostingRegressor(loss="quantile")` at
   two quantile levels on both rounds, exports them as `"ensemble": "boosting"`,
   and asserts runtime parity with scikit-learn.
3. `calibrate_v2.py` runs every (quantile, margin) candidate and RFC 6298 on the
   same validation transfers, then applies the pre-registered rule and writes
   `models/rto_model_v2.json` with the whole calibration table in its metadata.

`ForestModel.predict` in `timeout/ml_estimator.py` now handles both ensembles:
forest = mean of trees, boosting = `init + learning_rate x sum of trees`.

### 8.3 Live lab: `lab_server.py`

- `LabRunner` keeps one receiver process alive and runs one transfer at a time
  through `Lab.run_trial` (a fresh proxy process per run, same as the experiments).
- The sender's new `events(kind, details)` callback and the progress callback
  push lines into a `Run` object; `/api/events` streams them as Server-Sent
  Events and replays history to late subscribers.
- `parse_params` validates every field against fixed lists; `_static` only
  serves files inside `docs/site/`; the server binds to 127.0.0.1.
- The web page detects the server with `GET /api/health`. Without it (for
  example on the published page) the lab plays back recorded experiment runs.
