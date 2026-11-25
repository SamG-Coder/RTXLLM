RaytracingAccelerationStructure scene;
StructuredBuffer<uint> lengths;
RWStructuredBuffer<float> weights;
uint queries;

float gateWeight(uint matchLength)
{
    RayDesc ray;
    ray.Origin = float3(matchLength, 0, -1);
    ray.Direction = float3(0, 0, 1);
    ray.TMin = 0.001;
    ray.TMax = 2.1;
    RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
    q.TraceRayInline(scene, RAY_FLAG_NONE, 1, ray);
    while (q.Proceed()) {}
    return q.CommittedStatus() == COMMITTED_TRIANGLE_HIT ? q.CommittedRayT() - 1 : -1;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void trace(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x < queries) weights[tid.x] = gateWeight(lengths[tid.x]);
}

StructuredBuffer<int> dishIds;
StructuredBuffer<uint> offsets;
StructuredBuffer<uint> emissionBytes;
StructuredBuffer<float> emissionMass;
StructuredBuffer<float> transmission;
StructuredBuffer<int4> episode;
StructuredBuffer<uint> targetBytes;
uint layers, fullDistribution;

[shader("compute")]
[numthreads(64, 1, 1)]
void decode(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= queries) return;
    uint row = fullDistribution != 0 ? 0 : tid.x;
    uint target = targetBytes[tid.x];
    float p = 1.0 / 256.0;
    for (int s = int(layers) - 1; s >= 0; --s)
    {
        int dish = dishIds[row * layers + s];
        if (dish < 0) continue;
        uint begin = offsets[dish], end = offsets[dish + 1];
        uint lo = begin, hi = end;
        while (lo < hi)
        {
            uint mid = lo + (hi - lo) / 2;
            if (emissionBytes[mid] < target) lo = mid + 1;
            else hi = mid;
        }
        p *= transmission[dish];
        if (lo < end && emissionBytes[lo] == target) p += emissionMass[lo];
    }
    int4 hit = episode[row];
    float w = gateWeight(uint(hit.z));
    weights[tid.x] = w < 0 ? -1 : (1.0 - w) * p + (int(target) == hit.y ? w : 0.0);
}
