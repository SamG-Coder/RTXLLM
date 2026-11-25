RaytracingAccelerationStructure scene;
StructuredBuffer<uint> lengths;
RWStructuredBuffer<float> weights;
uint queries;

[shader("compute")]
[numthreads(64, 1, 1)]
void trace(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= queries) return;
    RayDesc ray;
    ray.Origin = float3(lengths[tid.x], 0, -1);
    ray.Direction = float3(0, 0, 1);
    ray.TMin = 0.001;
    ray.TMax = 2.1;
    RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
    q.TraceRayInline(scene, RAY_FLAG_NONE, 1, ray);
    while (q.Proceed()) {}
    weights[tid.x] = q.CommittedStatus() == COMMITTED_TRIANGLE_HIT ? q.CommittedRayT() - 1 : -1;
}
