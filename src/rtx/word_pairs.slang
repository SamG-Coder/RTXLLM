RaytracingAccelerationStructure scene;
StructuredBuffer<uint> older, recent, target, offsets, neighbours;
StructuredBuffer<float> energy, backoff;
RWStructuredBuffer<float4> readout;
uint vocabSize, queries;

float hit(uint x, uint y, out int row)
{
    RayDesc ray;
    ray.Origin = float3(x, y, 0);
    ray.Direction = float3(0, 0, 1);
    ray.TMin = 0;
    ray.TMax = 1.01;
    RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
    q.TraceRayInline(scene, RAY_FLAG_NONE, 0xff, ray);
    while (q.Proceed()) {}
    row = -1;
    if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT) return 0;
    row = int(q.CommittedPrimitiveIndex());
    return q.CommittedRayT();
}

[shader("compute")]
[numthreads(64, 1, 1)]
void trace(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= queries) return;
    uint a = older[i], b = recent[i], c = target[i];
    int row, unused;
    float strength = hit(a, b, row);
    float exact = (1-strength)*backoff[i];
    if (row >= 0) exact += strength*hit(c, vocabSize + uint(row), unused);
    float total = 0, weightedMass = 0;
    for (uint j = offsets[a]; j < offsets[a+1]; ++j)
    {
        hit(neighbours[j], b, row);
        if (row < 0) continue;
        total += energy[j];
        weightedMass += energy[j]*hit(c, vocabSize + uint(row), unused);
    }
    readout[i] = float4(exact, total > 0 ? weightedMass / total : exact, strength, total);
}
