RaytracingAccelerationStructure mirrors;
RaytracingAccelerationStructure receivers;
StructuredBuffer<float3> vertices;
StructuredBuffer<float3> origins;
RWStructuredBuffer<int2> hits;
uint rayCount;

int closest(RaytracingAccelerationStructure scene, RayDesc ray, out float t)
{
    RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
    q.TraceRayInline(scene, RAY_FLAG_NONE, 0xff, ray);
    while (q.Proceed()) {}
    t = 0;
    if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT) return -1;
    t = q.CommittedRayT();
    return int(q.CommittedPrimitiveIndex());
}

[shader("compute")]
[numthreads(64, 1, 1)]
void trace(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= rayCount) return;
    RayDesc r;
    r.Origin = origins[i];
    r.Direction = float3(0, 0, -1);
    r.TMin = 0.001;
    r.TMax = 512;
    float t;
    int tri = closest(mirrors, r, t);
    int2 result = int2(-1, -1);
    if (tri >= 0) {
        result.x = tri / 2;
        float3 a = vertices[3*tri];
        float3 n = normalize(cross(vertices[3*tri+1]-a, vertices[3*tri+2]-a));
        r.Origin += r.Direction*t;
        r.Direction = reflect(r.Direction, n);
        r.TMax = 4096;
        int readout = closest(receivers, r, t);
        if (readout >= 0) result.y = readout / 2;
    }
    hits[i] = result;
}
