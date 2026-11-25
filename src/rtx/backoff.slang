// #32: one ray per scored byte crosses the layered dish scene; every layer it meets
// is one backoff level (Claude). Layers are planes z = 1 + (k - m), m = k..0.
// A hit returns the dish (context) index; empty space at a layer means "unseen".
RaytracingAccelerationStructure scene;
StructuredBuffer<float2> points;        // [Q] lateral ray coordinates (dyadic, exact)
RWStructuredBuffer<int> dishes;         // [Q * layers], layer slot = k - m (top first)
uint queryCount, layers;
float startZ;                           // ray origin z (bounce budget = layers below it)

[shader("compute")]
[numthreads(64, 1, 1)]
void trace(uint3 tid : SV_DispatchThreadID)
{
    uint q = tid.x;
    if (q >= queryCount)
        return;
    for (uint s = 0; s < layers; ++s)
        dishes[q * layers + s] = -1;
    RayDesc ray;
    ray.Origin = float3(points[q], startZ);
    ray.Direction = float3(0, 0, 1);
    ray.TMin = 0.25;                     // layers are 1 apart: never re-hit the current plane
    ray.TMax = float(layers) + 2.0;
    float travelled = 0;
    for (uint hop = 0; hop < layers; ++hop)
    {
        RayQuery<RAY_FLAG_FORCE_OPAQUE> rq;
        rq.TraceRayInline(scene, RAY_FLAG_NONE, 0xFF, ray);
        while (rq.Proceed()) {}
        if (rq.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
            break;
        float t = rq.CommittedRayT();
        float z = ray.Origin.z + t;                     // world z of the hit layer
        int slot = int(round(z - 1.0));                 // 0 = most specific layer
        uint dish = rq.CommittedInstanceID() + rq.CommittedPrimitiveIndex() / 2;
        if (slot >= 0 && slot < int(layers))
            dishes[q * layers + slot] = int(dish);
        // Transmission: continue straight on from the hit plane.
        ray.Origin = float3(ray.Origin.xy, z);
    }
}
