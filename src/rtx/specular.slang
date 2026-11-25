// Each invocation owns its ray and its complete relation program.
RaytracingAccelerationStructure scene;
StructuredBuffer<float3> vertices;
StructuredBuffer<uint> kinds;
StructuredBuffer<int> owners;
StructuredBuffer<float3> stations;
StructuredBuffer<float3> directions;
StructuredBuffer<int> starts;
StructuredBuffer<int> programs;
RWStructuredBuffer<int> receivers;
uint queryCount;
uint programWidth;
float emissionOffset;
float tMin;

bool intersectLeg(RayDesc ray, uint kind, out uint primitive, out float distance)
{
    RayQuery<RAY_FLAG_FORCE_NON_OPAQUE> query;
    query.TraceRayInline(scene, RAY_FLAG_NONE, 255, ray);
    while (query.Proceed())
    {
        if (query.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE &&
            kinds[query.CandidatePrimitiveIndex()] == kind)
            query.CommitNonOpaqueTriangleHit();
    }
    if (query.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
    {
        primitive = 0;
        distance = 0;
        return false;
    }
    primitive = query.CommittedPrimitiveIndex();
    distance = query.CommittedRayT();
    return true;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void trace(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= queryCount)
        return;
    int station = starts[tid.x];
    uint row = tid.x * programWidth;
    for (uint hop = 0; hop < programWidth; ++hop)
    {
        int relation = programs[row + hop];
        if (relation < 0)
            break;
        RayDesc ray;
        ray.Direction = directions[relation];
        ray.Origin = stations[station] + emissionOffset * ray.Direction;
        ray.TMin = tMin;
        ray.TMax = 3.402823e38;
        uint mirror;
        float distance;
        if (!intersectLeg(ray, 1, mirror, distance))
            break;
        uint base = mirror * 3;
        float3 a = vertices[base];
        float3 normal = normalize(cross(vertices[base + 1] - a, vertices[base + 2] - a));
        ray.Origin += distance * ray.Direction;
        ray.Direction = reflect(ray.Direction, normal);

        uint receiver;
        if (!intersectLeg(ray, 0, receiver, distance))
            break;
        station = owners[receiver];
        receivers[row + hop] = station;
    }
}
