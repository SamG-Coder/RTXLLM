RaytracingAccelerationStructure scene;
StructuredBuffer<float2> xy;
StructuredBuffer<int> owners;
RWStructuredBuffer<int> rows;
uint count;

[shader("compute")]
[numthreads(64, 1, 1)]
void trace(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= count) return;
    RayDesc ray;
    ray.Origin = float3(xy[tid.x], 1);
    ray.Direction = float3(0, 0, -1);
    ray.TMin = 0.001;
    ray.TMax = 2;
    RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
    q.TraceRayInline(scene, RAY_FLAG_NONE, 1, ray);
    while (q.Proceed()) {}
    rows[tid.x] = q.CommittedStatus() == COMMITTED_TRIANGLE_HIT ?
        owners[q.CommittedInstanceID() + q.CommittedPrimitiveIndex()] : -1;
}
