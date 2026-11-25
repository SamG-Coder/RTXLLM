// ADR 0002 beam hop on RTX hardware (Claude, issue #11).
// One thread = one ray of one beam. Emission rays see only their relation's
// fact instance (mask bit 1 << (1 + r)); transport rays see only receivers (0x1).
RaytracingAccelerationStructure scene;
StructuredBuffer<float3> stations;    // [S]
StructuredBuffer<float3> directions;  // [R]
StructuredBuffer<float3> offsets;     // [R * raysPerBeam] world-space, core first
StructuredBuffer<float3> normals;     // [T] unit fact normals, sorted order
StructuredBuffer<int> owners;         // [T] receiver station, sorted order
StructuredBuffer<int> beamStation;    // [nb]
StructuredBuffer<int> beamRelation;   // [nb]
RWStructuredBuffer<int> landing;      // [nb * raysPerBeam]
uint beamCount, raysPerBeam;
float span, reach, transportRange, tMin;

bool closest(float3 o, float3 d, float tmax, uint mask, out uint tri, out float t)
{
    RayDesc ray;
    ray.Origin = o;
    ray.Direction = d;
    // CPU reference uses open interval (t_min, tmax); bump endpoints by one ulp.
    ray.TMin = asfloat(asuint(tMin) + 1);
    ray.TMax = max(ray.TMin, asfloat(asuint(tmax) - 1));
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
    uint i = tid.x;
    if (i >= beamCount * raysPerBeam)
        return;
    uint b = i / raysPerBeam;
    uint k = i - b * raysPerBeam;
    int r = beamRelation[b];
    float3 d = directions[r];
    float3 o = (stations[beamStation[b]] - span * d) + offsets[r * raysPerBeam + k];
    int result = -1;
    uint tri;
    float t;
    if (closest(o, d, reach, 1u << (1 + r), tri, t))
    {
        float3 p = o + t * d;
        float3 n = normals[tri];
        float3 dr = d - 2.0 * dot(d, n) * n;
        dr = dr / length(dr);
        if (closest(p, dr, transportRange, 1u, tri, t))
            result = owners[tri];
    }
    landing[i] = result;
}
