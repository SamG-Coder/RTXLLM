# Vulkan reflection smoke test

Run from the repository root with Python 3.12, an NVIDIA RTX GPU and a working Vulkan driver. SlangPy builds the Slang compute kernel at run time.

```powershell
py -3.12 -m venv "$env:TEMP/rtxllm-vk-env"
& "$env:TEMP/rtxllm-vk-env/Scripts/python" -m pip install slangpy==0.43.1
& "$env:TEMP/rtxllm-vk-env/Scripts/python" experiments/vulkan_smoke/run.py
```

On the development host, Vulkan adapter enumeration required disabling implicit layers for the test process. If adapter discovery fails, run the following in the test shell and retry.

```powershell
$env:VK_LOADER_LAYERS_DISABLE = '~implicit~'
```

A single shader invocation fires a ray along +Z. A tilted triangle reflects it along +X. The shader writes the committed primitive indices into two slots initialized to -1. The expected path is [0, 1]. Rebuilding the scene with the receiver displaced four units along Y changes the expected path to [0, -1]. Both paths must match for the program to exit successfully. The fixture measures hardware traversal and reflection only. Model accuracy and speed require separate experiments.
