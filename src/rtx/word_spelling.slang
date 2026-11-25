RaytracingAccelerationStructure scene;
RWStructuredBuffer<float> probabilities;
uint queries;
[shader("compute")]
[numthreads(64, 1, 1)]
void trace(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= queries) return;
    RayDesc r;
    r.Origin = float3(i % 257, i / 257, 0);
    r.Direction = float3(0, 0, 1);
    r.TMin = 0;
    r.TMax = 1.01;
    RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
    q.TraceRayInline(scene, RAY_FLAG_NONE, 0xff, r);
    while (q.Proceed()) {}
    probabilities[i] = q.CommittedStatus() == COMMITTED_TRIANGLE_HIT ? q.CommittedRayT() : 0;
}
