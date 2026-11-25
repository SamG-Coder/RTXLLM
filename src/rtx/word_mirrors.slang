RaytracingAccelerationStructure scene;
StructuredBuffer<float3> origins;
StructuredBuffer<float3> vertices;
RWStructuredBuffer<int3> paths;
uint count;
float3 incident;
float firstRange;

[shader("compute")]
[numthreads(64, 1, 1)]
void trace(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= count) return;
    RayDesc r;
    r.Origin = origins[i];
    r.Direction = incident;
    r.TMin = 0.001;
    r.TMax = firstRange;
    int3 path = int3(-1, -1, -1);
    for (uint depth = 0; depth < 3; ++depth) {
        RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
        q.TraceRayInline(scene, RAY_FLAG_NONE, 0xff, r);
        while (q.Proceed()) {}
        if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT) break;
        uint tri = q.CommittedPrimitiveIndex();
        path[depth] = int(tri / 6);
        float3 a = vertices[3 * tri];
        float3 n = normalize(cross(vertices[3*tri+1] - a, vertices[3*tri+2] - a));
        r.Origin += q.CommittedRayT() * r.Direction;
        r.Direction = reflect(r.Direction, n);
        r.TMax = 100.0;
    }
    paths[i] = path;
}
