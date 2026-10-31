# Project: AI-Adaptive Reliable File Transfer Protocol over UDP

## 1. Your Role

Act as a senior Computer Networks engineer, systems programmer, machine learning engineer, and technical mentor.

Help me design and build a complete, well-engineered Computer Networks project titled:

**"AI-Adaptive Reliable File Transfer Protocol over UDP Using Machine Learning-Based Retransmission Timeout Prediction."**

I am a Computer Science Engineering student specializing in Information Security. I want to understand the implementation deeply, not merely generate code that I cannot explain. Treat this as a serious academic project that I will demonstrate to my professor and defend during a project evaluation.

The final project must be functional, modular, well-documented, experimentally evaluated, and impressive enough to demonstrate strong knowledge of Computer Networks.

Do not give me a superficial prototype. Guide me through a structured implementation with clear explanations, working code, and measurable results.

## 2. Core Objective

Build a client-server file-transfer application using UDP sockets.

UDP does not inherently guarantee delivery, ordering, duplicate suppression, or retransmission. Therefore, implement a reliability layer on top of UDP that provides these properties.

The application must support:

1. Transferring files from a sender to a receiver over UDP.
2. Dividing files into numbered packets.
3. Detecting lost, corrupted, duplicated, and out-of-order packets.
4. Sending acknowledgements from the receiver to the sender.
5. Retransmitting packets when necessary.
6. Reassembling packets into the original file correctly.
7. Detecting corrupted packets using checksums or cryptographic hashes.
8. Dynamically adapting retransmission timeout values to changing network conditions.
9. Comparing a machine-learning-based timeout estimator with a conventional timeout estimator.
10. Displaying performance metrics through a command-line dashboard or a lightweight visualization.

The transferred file must be identical to the original file, verified using SHA-256 hashes.

## 3. Recommended Technology Stack

Use the following stack unless there is a compelling technical reason to change it:

- Language: Python 3.
- Networking: Python socket programming with UDP sockets.
- Machine learning: scikit-learn.
- Data processing: NumPy and Pandas, only where useful.
- Visualization: Matplotlib.
- Testing: pytest or Python's unittest.
- Network impairment simulation: Linux tc/netem where available, plus a portable Python-based packet impairment simulator for development and demonstrations.
- Version control: Git.

The application should run on macOS and Linux where practical. Explain any operating-system-specific limitations.

Keep the core networking implementation understandable and avoid unnecessary frameworks.

## 4. System Architecture

Design the project using separate modules:

### A. Sender
- Accept the source file path and receiver IP address and port.
- Read the file in configurable chunks.
- Assign a unique transfer ID and sequence number to each data packet.
- Maintain packet transmission timestamps and retransmission counts.
- Send packets over UDP.
- Track acknowledgements and outstanding packets.
- Detect timeout expiration.
- Retransmit packets when required.
- Record network observations for timeout estimation.
- Display transfer progress, throughput, retransmissions, and completion status.

### B. Receiver
- Listen on a UDP socket.
- Parse and validate incoming packets.
- Verify packet checksums.
- Handle duplicates and out-of-order arrivals.
- Buffer data until it can be reconstructed correctly.
- Send acknowledgements to the sender.
- Detect the end of the transfer.
- Reconstruct the file safely.
- Verify the final file using SHA-256.
- Send a completion confirmation.

### C. Reliable Transport Layer
Implement reliability independently of the file-reading and file-writing logic.

Include:
- Packet framing and serialization.
- Sequence numbers.
- Cumulative acknowledgements and, if appropriate, selective acknowledgements.
- A retransmission queue.
- Duplicate packet handling.
- Timeout handling.
- Transfer IDs.
- End-of-transfer signaling.
- Error handling and connection termination.
- Protection against indefinitely waiting for missing acknowledgements.

Choose a suitable protocol design, such as a selective-repeat sliding-window protocol. Explain why you selected it over stop-and-wait or go-back-N.

Use a bounded sliding window to allow multiple packets to be in flight simultaneously.

Implement modular sequence-number handling correctly if sequence numbers wrap around, or use a sufficiently large sequence-number space that makes wrapping practically irrelevant for the demonstration.

### D. AI-Based Retransmission Timeout Estimator

This is the main innovative component of the project.

Implement two timeout estimation strategies.

**Baseline A: Conventional adaptive timeout**

Use a standard TCP-inspired RTT estimator with:
- Smoothed RTT (SRTT).
- RTT variation (RTTVAR).
- Retransmission timeout calculation.
- Configurable minimum and maximum timeout values.
- Exponential backoff after repeated timeout events.

Use the standard Jacobson/Karels-style estimation approach and clearly explain its equations and parameters.

Do not treat RTT samples from retransmitted packets as unambiguous. Follow Karn's algorithm or an equivalent method to avoid incorrectly updating the RTT estimator.

**Baseline B: Machine-learning-based adaptive timeout**

Build an ML model that estimates a suitable retransmission timeout based on recent network behavior.

Start with a regression model such as RandomForestRegressor or another justified model. Compare alternatives only if doing so adds measurable value.

Potential features include:
- Recent RTT measurements.
- Smoothed RTT.
- RTT variation.
- Recent RTT trend.
- Recent packet loss rate.
- Recent retransmission count.
- Recent acknowledgement delays.
- Recent inter-arrival-time variation.
- Recent timeout history.
- Current timeout.
- Recent throughput or congestion indicators, where reliable measurements are available.

Do not blindly include every feature. Select features based on whether they are observable at the sender, meaningful, and available before the timeout decision.

The ML model must estimate a useful timeout based on past observations and must not use future information that would be unavailable at runtime.

Use an appropriate nonnegative target, such as a suitably chosen high quantile of future RTTs, with a defensible safety margin. Explain the target construction carefully.

Do not train the model using arbitrary labels or fabricated claims about accuracy.

### Important ML design requirements

1. Generate training data using repeatable network simulations and/or controlled experiments.
2. Record timestamps and network conditions.
3. Construct features from information available at prediction time.
4. Avoid data leakage between training and test sets.
5. Split training and test data by independent simulation runs or network scenarios when possible, rather than randomly splitting highly correlated adjacent samples.
6. Compare the ML estimator against the conventional estimator using the same network traces and transfer workloads.
7. Persist the trained model and load it during runtime.
8. Handle model startup, missing history, invalid predictions, and model-loading errors.
9. Enforce safe timeout bounds.
10. Retain a conventional estimator as a fallback.

**Critical requirement:** Do not assume that machine learning will outperform the conventional estimator. Design experiments to discover when it helps, when it does not, and why.

The ML component must operate online using available historical observations, but model training can happen offline. Explain this separation clearly.

### E. Timeout Decision and Safety Mechanism

The final timeout decision must be safe and bounded.

Use a design in which:
- The conventional estimator always remains available.
- The ML model produces a proposed timeout.
- The result is clipped to configurable minimum and maximum bounds.
- A safety policy prevents aggressive timeouts that would trigger excessive unnecessary retransmissions.
- The estimator falls back to the conventional value when the model is unavailable or its prediction is invalid.

Explain and justify the final combination policy. Evaluate whether the ML estimator should directly propose the timeout or predict a correction to the conventional estimator.

Avoid claiming that the timeout algorithm is superior until experimental results demonstrate it.

## 5. Packet Protocol Design

Design an explicit packet format with a header and payload.

Consider the following message types:
- SYN or transfer initialization.
- DATA.
- ACK.
- FIN.
- Completion acknowledgement.
- Optional error message.

Define appropriate fields, such as:
- Protocol version.
- Message type.
- Transfer ID.
- Sequence number.
- Acknowledgement number.
- Payload length.
- Checksum.
- Optional selective-acknowledgement information.

Specify the header format, field sizes, byte order, maximum datagram size, and validation rules.

Avoid sending large UDP datagrams that would be fragmented at the IP layer. Choose a conservative configurable payload size and explain the trade-off between packet overhead and fragmentation risk.

Clearly distinguish:
- UDP checksum behavior.
- Application-level packet-integrity checks.
- Final-file SHA-256 verification.

If the protocol uses a checksum for each packet, define exactly what bytes it covers and how it is calculated.

Ensure that a malicious or malformed datagram cannot cause uncontrolled memory allocation or unsafe file paths.

## 6. Network Conditions and Simulation

Build a reproducible test environment that can simulate:
- Low latency.
- High latency.
- Variable latency and jitter.
- Random packet loss.
- Burst packet loss.
- Packet reordering.
- Duplicate packets.
- A changing network path or network quality.
- Delayed acknowledgements.

Provide configurable network profiles, for example:
- Stable network.
- High-latency network.
- Lossy network.
- Jitter-heavy network.
- Rapidly changing network.

Clearly distinguish actual operating-system/network impairment from application-level simulation. Do not claim that application-level simulation perfectly reproduces a real network.

Use a portable Python-based impairment layer for development, and optionally explain how to run experiments using Linux tc/netem.

Make random experiments reproducible using recorded random seeds.

Ensure that the same impairment traces and equivalent transfer workloads are used when comparing timeout strategies.

## 7. Performance Metrics

Measure and report at least:

- Total file-transfer time.
- Effective throughput in bytes per second or Mbps.
- Number of packets sent.
- Number of retransmissions.
- Timeout events.
- Estimated and observed RTT.
- Unnecessary retransmissions caused by premature timeouts.
- Recovery time after packet loss.
- Packet loss rate.
- File-integrity verification result.
- Timeout-estimation error, where the target is appropriately defined.
- CPU and memory usage if practical.

Clearly define each metric and its calculation.

For throughput, distinguish useful file payload delivered from total network traffic.

For timeout-estimation evaluation, consider metrics such as MAE and the frequency of timeouts that expire before a delayed acknowledgement arrives. Explain why a lower prediction error does not automatically imply a better transport protocol.

Run multiple independent trials for each network profile and report means, variability, and relevant limitations.

Do not fabricate benchmark numbers. Generate real results from executed experiments.

## 8. Command-Line Interface

Create a clean command-line interface.

Example commands, to be adapted to the final implementation:

python receiver.py --host 0.0.0.0 --port 9000 --output-dir received_files

python sender.py --file sample.pdf --host 127.0.0.1 --port 9000 --timeout-mode rto

python sender.py --file sample.pdf --host 127.0.0.1 --port 9000 --timeout-mode ml

python train_model.py

python run_experiments.py

Provide useful options for:
- Payload size.
- Sliding-window size.
- Timeout bounds.
- Selected timeout strategy.
- Network simulation profile.
- Output directory.
- Logging level.

Make the commands accurate for the final project structure rather than leaving placeholder commands that do not run.

## 9. Project Structure

Propose a professional and modular directory structure, potentially resembling:

```
udp_ai_transfer/
    sender.py
    receiver.py
    protocol/
        packet.py
        reliability.py
        window.py
    timeout/
        conventional.py
        ml_estimator.py
        features.py
    simulation/
        impairment.py
        profiles.py
    experiments/
        run_experiments.py
        metrics.py
        report.py
    training/
        generate_dataset.py
        train_model.py
        evaluate_model.py
    tests/
        test_packet.py
        test_reliability.py
        test_timeout.py
        test_file_integrity.py
    data/
    models/
    received_files/
    logs/
    requirements.txt
    README.md
    .gitignore
```

Modify the structure when necessary, but maintain clear separation of concerns.

Do not create unnecessary abstractions or dozens of files just to make the project look sophisticated.

## 10. Testing and Correctness

Create automated tests for:
- Packet encoding and decoding.
- Invalid packet rejection.
- Corrupted payload detection.
- Missing packets.
- Duplicate packets.
- Out-of-order packets.
- Delayed ACKs.
- Retransmission behavior.
- Sliding-window operation.
- Timeout estimation.
- ML fallback behavior.
- Transfer completion.
- Empty files.
- Large files.
- Files containing arbitrary binary data.
- Receiver interruption and timeout recovery.
- Final-file SHA-256 verification.

Test both normal operation and adverse network conditions.

Prevent a transfer from silently reporting success if packets are missing or file integrity fails.

Define what happens if the sender or receiver crashes, a transfer ID is reused, or an acknowledgement is lost.

## 11. Security and Robustness

Although the focus is Computer Networks, incorporate basic security-conscious engineering.

Consider:
- Input validation.
- Safe output filenames.
- Transfer-ID validation.
- Datagram size limits.
- Bounded receive buffers.
- Timeouts and resource limits.
- Handling malformed packets.
- Preventing unbounded retransmission loops.
- Avoiding unsafe deserialization mechanisms such as pickle for untrusted network input.
- Optional HMAC-based packet authentication using a pre-shared key, if feasible.

Clearly state that checksums and SHA-256 integrity verification alone do not authenticate the sender. If HMAC is added, explain key management and replay considerations.

Do not present this academic UDP protocol as a replacement for production-grade secure transfer protocols.

## 12. Visual Dashboard and Demonstration

If practical, create a lightweight dashboard or a clear terminal-based live display that shows:
- Current transfer progress.
- Packets transmitted and acknowledged.
- Retransmissions.
- Current RTT.
- Conventional RTO.
- ML-proposed RTO.
- Final selected RTO.
- Current simulated network conditions.
- Throughput.
- Transfer-integrity status.

Generate plots comparing the two timeout strategies across the same experimental scenarios.

At minimum, produce:
1. Throughput comparison.
2. Retransmission-count comparison.
3. Transfer-duration comparison.
4. RTT and selected-timeout behavior over time.
5. Timeout-prediction evaluation.
6. Results across different network conditions.

Charts and reports must be generated from actual experimental outputs.

## 13. Implementation Workflow

Build the project incrementally in the following stages.

**Stage 1: Architecture and protocol design**
- Explain the architecture.
- Draw the sender-receiver workflow.
- Define the packet format.
- Explain the reliable transport design.
- Explain the ML problem formulation.
- Identify important design decisions and trade-offs.

**Stage 2: Minimal UDP file transfer**
- Implement a sender and receiver.
- Transfer a binary file.
- Verify the final SHA-256 hash.

**Stage 3: Reliability layer**
- Add sequence numbers, ACKs, retransmissions, and a sliding window.
- Add duplicate handling and out-of-order buffering.
- Test packet loss and delayed acknowledgements.

**Stage 4: Conventional timeout estimator**
- Implement RTT estimation and retransmission timeout calculation.
- Explain the equations.
- Add exponential backoff and correct RTT sampling.

**Stage 5: Network simulator**
- Implement configurable delay, jitter, loss, duplication, and reordering.
- Make tests reproducible.

**Stage 6: ML dataset and model**
- Generate realistic experimental observations.
- Define features and targets.
- Train and evaluate the model without leakage.
- Save and reload the model.

**Stage 7: AI-adaptive timeout integration**
- Integrate the estimator into the sender.
- Add bounds, fallback logic, and runtime logging.
- Compare both timeout modes.

**Stage 8: Evaluation and visualization**
- Run repeated experiments.
- Produce plots, tables, and a performance report.
- Interpret the actual findings honestly.

**Stage 9: Final polish**
- Add automated tests, documentation, setup instructions, error handling, and a clean demonstration workflow.

## 14. How I Want You to Work With Me

I want to understand the concepts and implement the project properly.

Follow these rules:

1. Start by giving me the architecture, packet format, protocol design, ML formulation, and implementation roadmap.
2. Explain why each major design decision is appropriate for this project.
3. Then implement the project stage by stage.
4. Provide complete code for each stage, including filenames and exact code placement.
5. Ensure modules integrate with each other and that imports, function names, arguments, and command-line options remain consistent.
6. Provide exact commands for installing dependencies, starting the receiver, sending a file, running tests, training the model, and executing experiments.
7. Explain important code sections in beginner-friendly language.
8. Explain relevant Computer Networks concepts, including UDP, RTT, RTO, ACKs, sequence numbers, sliding windows, packet loss, and retransmission.
9. When adding ML, explain the features, target variable, training data, inference process, and limitations.
10. Never pretend code has been executed or tested when it has not.
11. When a bug occurs, diagnose the underlying cause and provide a corrected, consistent solution.
12. Avoid unnecessary complexity, but do not remove essential reliability or evaluation features.
13. Do not generate all the code in one enormous, difficult-to-debug response.
14. Maintain a clear list of completed stages, remaining tasks, and important design decisions.
15. Include comments where they help understanding, but avoid explaining every obvious line.
16. If a design choice is uncertain, explain the alternatives and recommend one rather than silently making assumptions.

## 15. Final Deliverables

By the end, I want:

- A fully working UDP file-transfer application.
- A reliable transport layer implemented over UDP.
- A conventional adaptive RTO baseline.
- An ML-based adaptive timeout estimator.
- A reproducible network impairment simulator.
- A training and evaluation pipeline.
- Automated tests.
- Performance plots and experiment results.
- A professional README with setup and usage instructions.
- A protocol architecture diagram.
- A results table comparing both approaches.
- A project report structure.
- A concise presentation outline.
- Likely viva questions with accurate answers.
- A demonstration scenario showing normal transfer, packet loss, retransmission, and adaptive timeout behavior.

## 16. First Response: What You Must Do Now

Do not start by dumping the entire implementation.

In your first response, provide:

1. A clear explanation of the project in simple terms.
2. A professional system architecture diagram using Mermaid.
3. The end-to-end packet-transfer workflow.
4. The proposed packet header and its fields.
5. The conventional RTO equations.
6. The proposed ML prediction problem, including the training target and inference-time features.
7. How to compare the ML approach fairly against the conventional estimator.
8. The complete project directory structure.
9. A staged implementation checklist.
10. The recommended first implementation step.

Explain any important trade-offs and point out potential technical mistakes before they become implementation problems.

Prioritize correctness, reliability, explainability, reproducibility, and demonstrable experimental results over unnecessary complexity or flashy claims.

Let's build this project step by step, as a serious Computer Networks engineering project.
