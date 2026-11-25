RaytracingAccelerationStructure scene;
StructuredBuffer<int> owners;
RWStructuredBuffer<int> votes;
uint rowCount;
StructuredBuffer<uint> addresses;

[shader("compute")]
[numthreads(64, 1, 1)]
void trace(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    uint row = i / 256;
    if (row >= rowCount) return;
    uint address = addresses[row];
    RayDesc ray;
    ray.Origin = float3(float(address % 128) + (float(i % 256) + 0.5) / 256.0,
                        float(address / 128) + 0.375, 1);
    ray.Direction = float3(0, 0, -1);
    ray.TMin = 0.001;
    ray.TMax = 2;
    RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
    q.TraceRayInline(scene, RAY_FLAG_NONE, 1, ray);
    while (q.Proceed()) {}
    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
        votes[i] = owners[q.CommittedInstanceID() + q.CommittedPrimitiveIndex()];
    else
        votes[i] = -1;
}
