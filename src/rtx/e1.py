"""GPU tracer and parity runner for the E1 reference's NPZ format."""

import argparse
import json
from pathlib import Path

import numpy as np
import slangpy as sp
from run_specular import TriangleScene, gpu_buffer, open_kernel


class E1Scene(TriangleScene):
    def __init__(self, device, data):
        params = json.loads(str(data["params"]))
        super().__init__(
            device, dict(data, t_min=params["t_min"], emit_offset=params["emit_offset"])
        )
        if params["hop_mode"] not in {"specular", "transport"}:
            raise ValueError("Unknown hop mode")
        limit = params["max_events"]
        if not 0 <= limit <= 16:
            raise ValueError("E1 supports up to 16 events per lineage")
        disp = data["tri_displacement"]
        if disp.shape != (self.shape[0], 3) or disp.dtype != np.float32:
            raise ValueError("Expected float32[T,3] displacements")
        self.bindings.update(
            reflectance=gpu_buffer(device, data["tri_reflectance"][self.order]),
            displacement=gpu_buffer(device, disp[self.order]),
            stationCount=self.station_count,
            eventLimit=limit,
            transportMode=int(params["hop_mode"] == "transport"),
            openTransport=int(params["transport_hits_facts"]),
            energyFloor=params["energy_floor"],
            emissionRange=params["emit_range"],
            transportRange=params["transport_range"],
            arrivalRange=params["arrival_range"],
        )

    def trace(self, kernel, starts, programs):
        n, width = programs.shape
        if not n or not 1 <= width <= 16 or starts.shape != (n,):
            raise ValueError(
                "Expected nonempty program matrix of width 1..16 and matching start vector"
            )
        if starts.dtype != np.int32 or programs.dtype != np.int32:
            raise ValueError("Expected int32 starts and programs")
        if np.any((starts < 0) | (starts >= self.station_count)):
            raise ValueError("Invalid start")
        if np.any((programs < -1) | (programs >= self.relation_count)) or np.any(
            programs[:, 0] < 0
        ):
            raise ValueError("Invalid program")
        device = self.device
        output = {}
        for name, shape, dtype, initial in [
            ("answers", (n, self.station_count), np.float32, 0),
            ("lost", (n, 3), np.float32, 0),
            ("paths", (n, width + 1), np.int32, -2),
            ("counters", (n, 2), np.uint32, 0),
        ]:
            buf = gpu_buffer(
                device, np.full(shape, initial, dtype), sp.BufferUsage.unordered_access
            )
            output[name] = buf, dtype, shape
        bindings = {name: values[0] for name, values in output.items()}
        pool = device.create_query_pool(sp.QueryType.timestamp, 2)
        kernel.dispatch(
            thread_count=[n, 1, 1],
            vars=dict(
                self.bindings,
                **bindings,
                starts=gpu_buffer(device, starts),
                programs=gpu_buffer(device, programs),
                queryCount=n,
                programWidth=width,
            ),
            query_pool=pool,
            query_index_before=0,
            query_index_after=1,
        )
        device.wait_for_idle()
        a, b = pool.get_results(0, 2)
        result = {
            name: buf.to_numpy().view(dtype).reshape(shape)
            for name, (buf, dtype, shape) in output.items()
        }
        return result, dict(
            self.metrics,
            dispatch_gpu_ms=(b - a) * 1000 / device.info.timestamp_frequency,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixtures", nargs="+", type=Path)
    args = parser.parse_args()
    device, kernel = open_kernel("e1.slang")
    for fixture in args.fixtures:
        with np.load(fixture, allow_pickle=False) as f:
            scene = E1Scene(
                device,
                {
                    k: f[k]
                    for k in (
                        "params",
                        "station_pos",
                        "relation_dir",
                        "tri_kind",
                        "tri_vertices",
                        "tri_station",
                        "tri_displacement",
                        "tri_reflectance",
                    )
                },
            )
            result, metrics = scene.trace(kernel, f["start"], f["program"])
            energies = result["answers"]
            for key, got in [
                ("answer", energies),
                ("miss", result["lost"][:, 0]),
                ("floor", result["lost"][:, 1]),
                ("overflow", result["lost"][:, 2]),
            ]:
                np.testing.assert_allclose(got, f["expected_" + key], rtol=0, atol=1e-5)
            np.testing.assert_array_equal(result["paths"], f["expected_hops"])
            np.testing.assert_array_equal(
                np.where(energies.sum(1) >= 0.5, energies.argmax(1), -1),
                f["expected_pred"],
            )
            print(fixture.name, len(energies), "matched", metrics)


if __name__ == "__main__":
    main()
