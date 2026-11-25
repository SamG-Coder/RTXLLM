// #53 (Claude): first dish hit by each ray of a word-layout beam.
// Dishes are hexagonal fans (6 triangles) facing -d; one BLAS, identity TLAS.
RaytracingAccelerationStructure scene;
StructuredBuffer<float3> origins;     // [N]
RWStructuredBuffer<int> hitDish;      // [N] dish index, -1 = miss
uint rayCount;
float3 direction;
float tMin, tMax;

[shader("compute")]
[numthreads(64, 1, 1)]
void trace(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= rayCount)
        return;
    RayDesc ray;
    ray.Origin = origins[i];
    ray.Direction = direction;
    ray.TMin = asfloat(asuint(tMin) + 1);      // open interval, as in the CPU reference
    ray.TMax = asfloat(asuint(tMax) - 1);
    RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
    q.TraceRayInline(scene, RAY_FLAG_NONE, 0xFF, ray);
    while (q.Proceed()) {}
    hitDish[i] = q.CommittedStatus() == COMMITTED_TRIANGLE_HIT ? int(q.CommittedPrimitiveIndex() / 6) : -1;
}
