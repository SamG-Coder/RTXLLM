# RTXLLM

What if an LLM's knowledge existed as physical geometry in a simulated space, and inference happened by tracing rays and bouncing between pieces of information?

RTXLLM is an experimental research project exploring hardware ray tracing as a fundamental mechanism for spatial memory, learning and language inference. The aim is to investigate an alternative to conventional dense weight-based architectures, not simply accelerate an existing transformer or visualize a vector database.

Information is represented by spatial primitives; intersections and subsequent bounces contribute to the model's state and predictions. The exact representation and learning rules are open research questions.

## Development

Claude and Codex collaborate autonomously through GitHub issues, branches and mutual PR review. Both must agree on the exact revision before merging to main.

- [Research goal and milestones](GOAL.md)
- [Shared agent operating agreement](AGENTS.md)
- [Claude Code entry point](CLAUDE.md)

## Status

- Multi-hop retrieval composes written facts on RTX and matches the CPU reference. [Hardware parity](src/rtx/README.md)
- The first byte predictors encode quantized n-gram distributions in routing mirrors. Learning eight shared mirror routes improved on frozen and shuffled controls across three seeds on Alice, while a conventional bigram predicted better. [Learned routing results](experiments/rtx_credit/README.md)
- The optical cavity reached 2.795 bits per byte on held-out Dracula text. The RTX implementation of modified Kneser-Ney reproduced the six-gram score of 2.017 on the same targets. [Cavity experiment](https://github.com/SamG-Coder/RTXLLM/pull/25) and [RTX count reference](experiments/e8_hardware_backoff/README.md)
- Ongoing experiments test generalization to unwritten facts and next-byte prediction from recent sequences. A learned copy gate improved the five-gram reference from 2.219 to 2.201 bits per byte on The War of the Worlds. [Shared-geometry study](https://github.com/SamG-Coder/RTXLLM/issues/11) and [episodic integration under review](https://github.com/SamG-Coder/RTXLLM/pull/41)

## Train the one-byte-context mirror learner on local files

These commands train the experimental E4 model from [the shared routing study](experiments/rtx_credit/README.md).

These PowerShell commands run from the repository root with an NVIDIA RTX GPU and a Vulkan driver. Training streams raw file bytes. Generation traces the learned geometry for every sampled byte.

```powershell
py -3.12 -m venv "$env:TEMP/rtxllm-vk-env"
$python = "$env:TEMP/rtxllm-vk-env/Scripts/python"
& $python -m pip install slangpy==0.43.1
$env:VK_LOADER_LAYERS_DISABLE = '~implicit~'  # Needed for adapter discovery on the development host
& $python tools/byte_model.py train README.md model-a --seed 0
& $python tools/byte_model.py update model-a GOAL.md model-b
& $python tools/byte_model.py generate model-b/scene.npz "Geometry" --count 64 --seed 7
```

Each training run publishes a new directory after verifying the reloaded geometry. Keep both files to resume training. The geometry checkpoint, `scene.npz`, is sufficient for generation. The separate `training.npz` holds accumulated byte counts, optimizer coordinates and the random generator state. Updates ordinarily start a new document. Use `--continuation` when the new file follows the previous file byte for byte. The generator prints escaped bytes by default. Use `--out sample.bin` to save the prompt and sampled bytes to a new binary file.

## License

MIT.
