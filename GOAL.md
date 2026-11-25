# Goal: RTX-native language intelligence

Investigate whether language learning, memory and inference can emerge from information represented as persistent spatial geometry and queried through hardware-accelerated ray tracing and multiple bounces. Information must have an actual location in a simulated scene, with stable identity and inspectable payload; related information should be reachable through meaningful spatial paths. This is a research hypothesis, not a demonstrated LLM.

## Research direction
- Encode bytes, tokens, facts or learned features as spatial primitives with payloads, orientation and material-like interaction parameters. Explore learned placement, routing and scattering rather than assume ordinary Euclidean proximity captures language.
- Encode input into ray origins, directions and payloads. Use RTX acceleration structures and real intersections; let hits alter subsequent rays and accumulate evidence. Bound bounce depth and energy, handle misses, cycles and contradictions.
- Explore learning by adding or modifying geometry, payloads and interaction parameters from data and prediction error. Persist and reload memory without losing identity.
- Develop a native next-byte/token prediction mechanism driven by traced evidence. Existing models may serve as comparison baselines, not as a hidden answer generator. Retrieval alone is not completion of the mission.
- Measure whether additional bounces actually improve multi-hop reasoning and prediction. Compare with direct lookup, zero/one-bounce, randomized geometry, graph traversal, vector retrieval and a small conventional language model.

## Milestones
1. Agree on architecture, representation, hardware backend (evaluate OptiX versus Vulkan RT), data format and reproducible evaluation protocol.
2. Build a deterministic CPU reference and tiny synthetic scene with known multi-hop answers, distractors, misses and cycles.
3. Implement real RTX traversal and bounce logic, verify parity with the reference, report GPU timings and memory/build/update overhead.
4. Ingest raw text, learn spatial associations, support incremental updates and persistence; test held-out queries without answer leakage.
5. Generate next-byte/token distributions from traced state; measure held-out loss, perplexity where appropriate, generation quality, throughput and memory.
6. Scale corpus and scene size, profile bottlenecks and publish reproducible positive and negative findings.

## Proposed structure (agents may revise by agreement)
src/core/ — scene, payloads and interfaces
src/cpu/ — reference traversal
src/rtx/ — hardware traversal and shaders
src/learning/ — ingestion and adaptation
src/inference/ — query encoding and decoding
tests/ — correctness and parity
benchmarks/ — baselines, datasets and measurements
docs/decisions/ — architecture decisions
experiments/ — reproducible research

## Definition of success
A reproducible prototype that learns from supplied data, stores information spatially, uses genuine RTX intersections and multi-bounce paths to influence language predictions, and reports measured quality and cost against fair baselines. No speedup or superiority is assumed. Claude and Codex jointly choose implementation details, maintain issues and review each other's work under AGENTS.md.
