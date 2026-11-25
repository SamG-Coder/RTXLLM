RaytracingAccelerationStructure scene;
StructuredBuffer<float3> vertices;
RWStructuredBuffer<int> path;

[shader("compute")]
[numthreads(1, 1, 1)]
void trace()
{
    RayDesc ray;
    ray.Origin = float3(0, 0, -2);
    ray.Direction = float3(0, 0, 1);
    ray.TMin = 0.001;
    ray.TMax = 8;
    for (int depth = 0; depth < 2; ++depth)
    {
        RayQuery<RAY_FLAG_FORCE_OPAQUE> query;
        query.TraceRayInline(scene, RAY_FLAG_NONE, 255, ray);
        while (query.Proceed()) {}
        if (query.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
            break;
        uint primitive = query.CommittedPrimitiveIndex();
        path[depth] = int(primitive);
        uint base = 3 * primitive;
        float3 a = vertices[base];
        float3 normal = normalize(cross(vertices[base + 1] - a, vertices[base + 2] - a));
        ray.Origin += query.CommittedRayT() * ray.Direction;
        ray.Direction = reflect(ray.Direction, normal);
    }
}
