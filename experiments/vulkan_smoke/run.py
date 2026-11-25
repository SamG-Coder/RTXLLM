"""Run the two-intersection geometry control on a physical NVIDIA RTX adapter."""

from pathlib import Path

import numpy as np
import slangpy as sp


def build(device, geometry, kind):
    desc = sp.AccelerationStructureBuildDesc({"inputs": [geometry]})
    sizes = device.get_acceleration_structure_sizes(desc)
    structure = device.create_acceleration_structure(
        kind=kind, size=sizes.acceleration_structure_size
    )
    scratch = device.create_buffer(
        size=sizes.scratch_size, usage=sp.BufferUsage.unordered_access
    )
    commands = device.create_command_encoder()
    commands.build_acceleration_structure(
        desc=desc, dst=structure, src=None, scratch_buffer=scratch
    )
    device.submit_command_buffer(commands.finish())
    device.wait_for_idle()
    return structure


def trace_scene(device, kernel, receiver_offset):
    triangles = np.array([
        [-1, -1, -1], [1, -1, 1], [0, 1, 0],
        [2, -1, -1], [2, 1, -1], [2, 0, 1],
    ], dtype=np.float32)
    triangles[3:, 1] += receiver_offset
    vertices = device.create_buffer(
        data=triangles,
        usage=sp.BufferUsage.acceleration_structure_build_input | sp.BufferUsage.shader_resource,
    )
    geometry = sp.AccelerationStructureBuildInputTriangles({
        "vertex_buffers": [vertices],
        "vertex_count": len(triangles),
        "vertex_stride": triangles.strides[0],
        "vertex_format": sp.Format.rgb32_float,
        "flags": sp.AccelerationStructureGeometryFlags.opaque,
    })
    blas = build(device, geometry, sp.AccelerationStructureKind.bottom_level)
    instances = device.create_acceleration_structure_instance_list(1)
    instances.write(0, {
        "transform": sp.float3x4.identity(),
        "instance_mask": 255,
        "acceleration_structure": blas.handle,
    })
    scene = build(device, instances.build_input_instances(), sp.AccelerationStructureKind.top_level)
    path = device.create_buffer(data=np.full(2, -1, np.int32), usage=sp.BufferUsage.unordered_access)
    kernel.dispatch(thread_count=[1, 1, 1], vars={"scene": scene, "vertices": vertices, "path": path})
    return path.to_numpy().view(np.int32).tolist()


def main():
    adapters = sp.Device.enumerate_adapters(sp.DeviceType.vulkan)
    adapter = next((a for a in adapters if a.vendor_id == 0x10DE and "RTX" in a.name), None)
    if adapter is None:
        raise RuntimeError(f"An NVIDIA RTX Vulkan adapter is required, found {adapters}")
    device = sp.Device(
        type=sp.DeviceType.vulkan, adapter_luid=adapter.luid,
        compiler_options={"include_paths": [Path(__file__).resolve().parent]},
    )
    for feature in (sp.Feature.hardware_device, sp.Feature.acceleration_structure, sp.Feature.ray_query):
        if not device.has_feature(feature):
            raise RuntimeError(f"{adapter.name} lacks {feature}")
    kernel = device.create_compute_kernel(device.load_program("bounce.slang", ["trace"]))
    print(f"Vulkan | {adapter.name}")
    for offset, expected in [(0, [0, 1]), (4, [0, -1])]:
        actual = trace_scene(device, kernel, offset)
        print(f"receiver y offset {offset}: {actual}")
        if actual != expected:
            raise RuntimeError(f"Expected {expected}, observed {actual}")


if __name__ == "__main__":
    main()
