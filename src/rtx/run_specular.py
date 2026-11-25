"""Opaque specular path prototype. Accepts scene.npz and queries.npz."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import slangpy as sp


def gpu_buffer(device, array, usage=sp.BufferUsage.shader_resource):
    return device.create_buffer(data=np.ascontiguousarray(array), usage=usage)


class UpdatableStructure:
    def __init__(self, device, geometry, kind):
        self.desc = sp.AccelerationStructureBuildDesc(
            {
                "inputs": [geometry],
                "flags": sp.AccelerationStructureBuildFlags.allow_update,
            }
        )
        self.sizes = device.get_acceleration_structure_sizes(self.desc)
        self.structure = device.create_acceleration_structure(
            kind=kind, size=self.sizes.acceleration_structure_size
        )
        self.scratch = device.create_buffer(
            size=max(self.sizes.scratch_size, self.sizes.update_scratch_size),
            usage=sp.BufferUsage.unordered_access,
        )
        self.submit(device, update=False)

    def submit(self, device, *, update):
        self.desc.mode = (
            sp.AccelerationStructureBuildMode.update
            if update
            else sp.AccelerationStructureBuildMode.build
        )
        commands = device.create_command_encoder()
        commands.build_acceleration_structure(
            desc=self.desc,
            dst=self.structure,
            src=self.structure if update else None,
            scratch_buffer=self.scratch,
        )
        device.submit_command_buffer(commands.finish())
        device.wait_for_idle()


class SpecularScene:
    def __init__(self, device, scene):
        # These arrays bind directly to shader buffers, so dtype and shape are ABI.
        vertices = scene["tri_vertices"]
        if vertices.ndim != 3 or vertices.shape[1:] != (3, 3) or not len(vertices):
            raise ValueError("Expected nonempty triangle list (T, 3, 3)")
        for key, dtype, shape in [
            ("tri_vertices", np.float32, vertices.shape),
            ("tri_station", np.int32, (len(vertices),)),
            ("tri_kind", np.uint32, (len(vertices),)),
            ("tri_reflectance", np.float32, (len(vertices),)),
            ("station_pos", np.float32, (len(scene["station_pos"]), 3)),
            ("relation_dir", np.float32, (len(scene["relation_dir"]), 3)),
        ]:
            if scene[key].dtype != dtype or scene[key].shape != shape:
                raise ValueError(f"{key} requires {dtype} with shape {shape}")
        if not np.all(scene["tri_reflectance"][scene["tri_kind"] == 1] == 1):
            raise ValueError("This prototype supports fully reflective facts only")
        owners = scene["tri_station"][scene["tri_kind"] == 0]
        if np.any((owners < 0) | (owners >= len(scene["station_pos"]))):
            raise ValueError("Invalid receiver owner")
        self.vertices = gpu_buffer(
            device,
            vertices,
            sp.BufferUsage.shader_resource
            | sp.BufferUsage.acceleration_structure_build_input,
        )
        geometry = sp.AccelerationStructureBuildInputTriangles(
            {
                "vertex_buffers": [self.vertices],
                "vertex_count": len(vertices) * 3,
                "vertex_format": sp.Format.rgb32_float,
                "vertex_stride": 12,
            }
        )
        self.device = device
        self.shape = vertices.shape
        self.station_count = len(scene["station_pos"])
        self.relation_count = len(scene["relation_dir"])
        build_start = perf_counter()
        self.blas = UpdatableStructure(
            device, geometry, sp.AccelerationStructureKind.bottom_level
        )
        self.instances = device.create_acceleration_structure_instance_list(1)
        self.instances.write(
            0,
            {
                "transform": sp.float3x4.identity(),
                "instance_mask": 255,
                "acceleration_structure": self.blas.structure.handle,
            },
        )
        self.tlas = UpdatableStructure(
            device,
            self.instances.build_input_instances(),
            sp.AccelerationStructureKind.top_level,
        )
        self.metrics = {
            "as_build_wall_ms": (perf_counter() - build_start) * 1000,
            "as_bytes": self.blas.sizes.acceleration_structure_size
            + self.tlas.sizes.acceleration_structure_size,
            "resident_scratch_bytes": self.blas.scratch.size + self.tlas.scratch.size,
        }
        self.bindings = {
            "scene": self.tlas.structure,
            "vertices": self.vertices,
            "kinds": gpu_buffer(device, scene["tri_kind"]),
            "owners": gpu_buffer(device, scene["tri_station"]),
            "stations": gpu_buffer(device, scene["station_pos"]),
            "directions": gpu_buffer(device, scene["relation_dir"]),
            "tMin": float(scene["t_min"]),
            "emissionOffset": float(scene["emit_offset"]),
        }

    def refit_vertices(self, vertices):
        """Update positions with the same primitive count, order and ownership."""
        if vertices.dtype != np.float32 or vertices.shape != self.shape:
            raise ValueError("Refits require the original float32 geometry shape")
        begin = perf_counter()
        self.vertices.copy_from_numpy(np.ascontiguousarray(vertices))
        self.blas.submit(self.device, update=True)
        self.tlas.submit(self.device, update=True)
        return (perf_counter() - begin) * 1000

    def trace(self, kernel, queries):
        device = self.device
        starts, programs = queries["start"], queries["program"]
        if (
            starts.ndim != 1
            or programs.ndim != 2
            or starts.dtype != np.int32
            or programs.dtype != np.int32
        ):
            raise ValueError("Expected int32 start vector and program matrix")
        n, width = programs.shape
        if n != len(starts) or min(n, width) == 0:
            raise ValueError("Expected one start per nonempty program row")
        if np.any((starts < 0) | (starts >= self.station_count)):
            raise ValueError("Start station out of range")
        if np.any((programs < -1) | (programs >= self.relation_count)):
            raise ValueError("Relation out of range")
        output = gpu_buffer(
            device, np.full((n, width), -1, np.int32), sp.BufferUsage.unordered_access
        )
        bindings = {
            **self.bindings,
            "starts": gpu_buffer(device, starts),
            "programs": gpu_buffer(device, programs),
            "receivers": output,
            "queryCount": n,
            "programWidth": width,
        }
        pool = device.create_query_pool(sp.QueryType.timestamp, 2)
        kernel.dispatch(
            thread_count=[n, 1, 1],
            vars=bindings,
            query_pool=pool,
            query_index_before=0,
            query_index_after=1,
        )
        device.wait_for_idle()
        before, after = pool.get_results(0, 2)
        metrics = dict(
            self.metrics,
            dispatch_gpu_ms=(after - before) * 1000 / device.info.timestamp_frequency,
        )
        return output.to_numpy().view(np.int32).reshape(n, width), metrics


def open_kernel():
    adapter = next(
        (
            a
            for a in sp.Device.enumerate_adapters(sp.DeviceType.vulkan)
            if a.vendor_id == 0x10DE and "RTX" in a.name
        ),
        None,
    )
    if adapter is None:
        raise RuntimeError("An NVIDIA RTX Vulkan adapter is required")
    device = sp.Device(
        type=sp.DeviceType.vulkan,
        adapter_luid=adapter.luid,
        compiler_options={"include_paths": [Path(__file__).resolve().parent]},
    )
    for feature in (
        sp.Feature.hardware_device,
        sp.Feature.ray_query,
        sp.Feature.timestamp_query,
    ):
        if not device.has_feature(feature):
            raise RuntimeError(f"{adapter.name} lacks {feature}")
    kernel = device.create_compute_kernel(
        device.load_program("specular.slang", ["trace"])
    )
    print(adapter.name)
    return device, kernel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("queries", type=Path)
    parser.add_argument("output", type=Path, help="Receiver sequence output (.npy)")
    args = parser.parse_args()
    device, kernel = open_kernel()
    with np.load(args.scene) as scene, np.load(args.queries) as queries:
        receivers, metrics = SpecularScene(device, scene).trace(kernel, queries)
    np.save(args.output, receivers)
    print(metrics)


if __name__ == "__main__":
    main()
