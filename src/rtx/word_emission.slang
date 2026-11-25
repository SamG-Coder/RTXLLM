RaytracingAccelerationStructure scene;
StructuredBuffer<uint> previous, following, offsets, neighbours;
StructuredBuffer<float> energy;
RWStructuredBuffer<float2> result;
uint vocabSize, queryCount;

float mass(uint column, uint row)
{
    RayDesc ray;
    ray.Origin = float3(column, row, 0);
    ray.Direction = float3(0, 0, 1);
    ray.TMin = 0;
    ray.TMax = 1.01;
    RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
    q.TraceRayInline(scene, RAY_FLAG_NONE, 0xff, ray);
    while (q.Proceed()) {}
    return q.CommittedStatus() == COMMITTED_TRIANGLE_HIT ? q.CommittedRayT() : 0;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void trace(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= queryCount) return;
    uint src = previous[i], dest = following[i];
    uint knRow = vocabSize + src;
    float prior = mass(dest, knRow) + mass(vocabSize, knRow) * mass(dest, 2 * vocabSize);
    uint begin = offsets[src], end = offsets[src + 1];
    float beam = 0;
    for (uint j = begin; j < end; j++)
        beam += energy[j] * mass(dest, neighbours[j]);
    result[i] = float2(prior, begin == end ? prior : 0.9 * prior + 0.1 * beam);
}
