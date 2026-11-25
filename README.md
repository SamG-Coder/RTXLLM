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

The experimental byte learner trains routing mirrors from two-reflection RTX prediction error. Each input byte selects a launch station. Its mirror routes a ray to one of eight shared receivers, and continuation mirrors at the receiver vote for the next byte. The model conditions on one byte of history. The E4 evaluation reports 4.63–4.96 held-out bits/byte on *Alice’s Adventures in Wonderland* across three seeds. An unrestricted bigram scored 3.33 on the same targets.

- [Training experiment and limitations](experiments/rtx_credit/README.md)
- [Two-byte context routing experiment](experiments/two_reflection_bytes/README.md)
- [RTX multi-hop retrieval and parity](src/rtx/README.md)

## Train on local bytes

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
