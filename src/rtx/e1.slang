RaytracingAccelerationStructure scene;
StructuredBuffer<float3> vertices, stations, directions, displacement;
StructuredBuffer<uint> kinds;
StructuredBuffer<int> owners, starts, programs;
StructuredBuffer<float> reflectance;
RWStructuredBuffer<float> answers;
RWStructuredBuffer<float3> lost;
RWStructuredBuffer<int> paths;
RWStructuredBuffer<uint2> counters;

uint queryCount, stationCount, programWidth, eventLimit, transportMode, openTransport;
float tMin, emissionOffset, emissionRange, transportRange, arrivalRange, energyFloor;

// DFS needs at most one deferred transmission per intersection on the live
// lineage. The host validates both bounded dimensions before dispatch.
struct Branch
{
    float3 origin, direction;
    float reach, energy;
    uint events, hop, emission;
    int path[17];
};

void emitFrom(inout Branch b, int station, int relation)
{
    b.direction = directions[relation];
    b.origin = stations[station] + emissionOffset * b.direction;
    b.reach = emissionRange;
    b.emission = 1;
}

[shader("compute")]
[numthreads(32, 1, 1)]
void trace(uint3 tid : SV_DispatchThreadID)
{
    uint q = tid.x;
    if (q >= queryCount) return;
    uint row = q * programWidth;
    Branch b = {};
    b.path[0] = starts[q];
    b.energy = 1;
    emitFrom(b, starts[q], programs[row]);
    Branch deferred[16] = {};
    uint pending = 0;
    float bestEnergy = -1;
    uint bestDepth = 0;
    float3 loss = 0;
    uint rays = 0, crossTalk = 0;
    while (true)
    {
        RayDesc ray;
        ray.Origin = b.origin;
        ray.Direction = b.direction;
        // The reference uses open interval endpoints; Vulkan accepts endpoints.
        ray.TMin = asfloat(asuint(tMin) + 1);
        ray.TMax = max(ray.TMin, asfloat(asuint(max(b.reach, tMin)) - 1));
        RayQuery<RAY_FLAG_FORCE_OPAQUE> rq;
        uint mask = b.emission != 0 || openTransport != 0 ? 3 : 1;
        rq.TraceRayInline(scene, RAY_FLAG_NONE, mask, ray);
        while (rq.Proceed()) {}
        ++rays;
        bool finished = false;
        if (rq.CommittedStatus() != COMMITTED_TRIANGLE_HIT || rq.CommittedRayT() >= b.reach)
        {
            loss.x += b.energy;
            b.path[++b.hop] = -1;
            finished = true;
        }
        else if (++b.events > eventLimit)
            loss.z += b.energy;
        else
        {
            uint tri = rq.CommittedInstanceID() + rq.CommittedPrimitiveIndex();
            float t = rq.CommittedRayT();
            b.origin += t * b.direction;
            if (kinds[tri] == 0)
            {
                int station = owners[tri];
                b.path[++b.hop] = station;
                if (b.hop < programWidth && programs[row + b.hop] >= 0)
                {
                    emitFrom(b, station, programs[row + b.hop]);
                    continue;
                }
                answers[q * stationCount + station] += b.energy;
                finished = true;
            }
            else
            {
                if (b.emission == 0) ++crossTalk;
                float transmitted = b.energy * (1 - reflectance[tri]);
                if (transmitted > 0 && transmitted >= energyFloor)
                {
                    Branch pass = b;
                    pass.energy = transmitted;
                    pass.reach -= t;
                    deferred[pending++] = pass;
                }
                else loss.y += transmitted;
                b.energy *= reflectance[tri];
                if (b.energy > 0 && b.energy >= energyFloor)
                {
                    b.emission = 0;
                    if (transportMode != 0)
                    {
                        b.direction = normalize(displacement[tri]);
                        b.origin += displacement[tri];
                        b.reach = arrivalRange;
                    }
                    else
                    {
                        uint base = 3 * tri;
                        float3 a = vertices[base];
                        float3 n = normalize(cross(vertices[base + 1] - a, vertices[base + 2] - a));
                        b.direction = normalize(reflect(b.direction, n));
                        b.reach = transportRange;
                    }
                    continue;
                }
                loss.y += b.energy;
            }
        }
        // CPU's breadth-first reference breaks equal-energy ties by earliest
        // terminal frontier, then follow-before-transmit order on that frontier.
        uint depth = b.events + (finished && b.path[b.hop] == -1 ? 1 : 0);
        if (finished && (b.energy > bestEnergy || (b.energy == bestEnergy && depth < bestDepth)))
        {
            bestEnergy = b.energy;
            bestDepth = depth;
            for (uint h = 0; h <= programWidth; ++h)
                paths[q * (programWidth + 1) + h] = h <= b.hop ? b.path[h] : -2;
        }
        if (pending == 0) break;
        b = deferred[--pending];
    }
    lost[q] = loss;
    counters[q] = uint2(rays, crossTalk);
}
