# E1 on Vulkan RTX

Run with Python 3.12, NumPy, SlangPy 0.43.1 and a physical NVIDIA RTX adapter. The fixtures and integration benchmark use the CPU contract in ADR 0001. Each query runs in one compute invocation with a depth-first stack. Separate receiver and fact instances allow the ray mask to cull entire primitive classes in hardware. Emission has finite reach. Branches below the energy floor and branches exceeding the event cap are accounted for separately. The implementation accepts programs and event budgets up to 16.

From the repository root, in the environment created for the Vulkan smoke test:

```powershell
$env:VK_LOADER_LAYERS_DISABLE = '~implicit~'
$env:VK_INSTANCE_LAYERS = 'VK_LAYER_KHRONOS_validation'
$env:PYTHONPATH = (Resolve-Path src).Path
& "$env:TEMP/rtxllm-vk-env/Scripts/python" -B src/rtx/check_specular.py
& "$env:TEMP/rtxllm-vk-env/Scripts/python" -B src/rtx/e1.py tests/fixtures/e1_small_specular.npz tests/fixtures/e1_small_transport.npz
& "$env:TEMP/rtxllm-vk-env/Scripts/python" -B src/rtx/compare_e1.py --seed 2 --masked --stress-caps
foreach ($seed in 0..2) {
    & "$env:TEMP/rtxllm-vk-env/Scripts/python" -B src/rtx/compare_e1.py --split test --radius 0.05 --controls --seed $seed
}
```

The runner defaults to validation. It checks dominant receiver sequences, predicted stations, ray counts, off-emission fact hits, answer energies and all loss channels against fresh CPU traces. The stress run uses an energy floor of 0.6 and an event cap of 3. Both modes matched with 184 units of floor energy and 638 units of overflow across 1,000 validation queries.

## Held-out results

RTX 5080, driver 616.64, Windows, primary radius 0.05, open transport. Each seed has 1,000 queries, including 300 unique paths of each length. Traced results matched the independent CPU reference for full, one-hop and shuffled runs in each mode. The 30 preregistered E1 checks passed in each mode. Lower specular accuracy on some paths reflects physical interceptions observed by both implementations.

| Seed | Specular L2 | Specular L3 | Transport L2 / L3 | Specular GPU ms | Transport GPU ms |
|---|---:|---:|---:|---:|---:|
| 0 | 1.0000 | 0.9867 | 1.0000 / 1.0000 | 0.0900 | 0.0805 |
| 1 | 0.9967 | 0.9933 | 1.0000 / 1.0000 | 0.0903 | 0.0826 |
| 2 | 0.9900 | 0.9833 | 1.0000 / 1.0000 | 0.0789 | 0.0720 |

GPU ms measures the median dispatch for a batch of 1,000 queries after one warm-up, over five repeats. Full host batch time was 12.0–15.1 ms, including per-batch allocations, upload, synchronization and output readback. The three scenes required 761,856–766,592 bytes of acceleration structures and 54,272–54,528 bytes of resident build/update scratch. These figures exclude vertex/material/query/output buffers. The NumPy reference took 2.02–2.17 seconds per batch with brute-force triangle intersection. Symbolic graph traversal and optimized CPU BVHs were not timed in this comparison.

The older opaque path pilot remains available for existing byte checkpoints. Its schema uses distinct fact and receiver legs. Use the E1 tracer for ADR 0001 scenes and energy accounting.
