RaytracingAccelerationStructure scene;
StructuredBuffer<float3> vertices;
StructuredBuffer<float3> directions;
StructuredBuffer<uint> history;
RWStructuredBuffer<float2> landings;
uint queryCount, width;

bool hit(RayDesc ray, uint mask, out uint tri, out float t)
{
    RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
    q.TraceRayInline(scene, RAY_FLAG_NONE, mask, ray);
    while (q.Proceed()) {}
    if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
    {
        tri = 0;
        t = 0;
        return false;
    }
    tri = q.CommittedInstanceID() + q.CommittedPrimitiveIndex();
    t = q.CommittedRayT();
    return true;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void trace(uint3 tid : SV_DispatchThreadID)
{
    uint row = tid.x;
    if (row >= queryCount) return;
    RayDesc ray;
    ray.Origin = float3(0, 0, 0);
    ray.TMin = 0.0001;
    ray.TMax = 40;
    for (uint i = 0; i < width; ++i)
    {
        ray.Direction = directions[history[row * width + i]];
        uint mirror;
        float t;
        if (!hit(ray, 2, mirror, t))
        {
            landings[row] = float2(1000, 1000);
            return;
        }
        ray.Origin += t * ray.Direction;
        uint base = 3 * mirror;
        float3 a = vertices[base];
        float3 n = normalize(cross(vertices[base+1]-a, vertices[base+2]-a));
        ray.Direction = reflect(ray.Direction, n);
        uint plane;
        if (!hit(ray, 1, plane, t))
        {
            landings[row] = float2(1000, 1000);
            return;
        }
        ray.Origin += t * ray.Direction;
    }
    landings[row] = ray.Origin.xy;
}
