RaytracingAccelerationStructure scene;
RWStructuredBuffer<int4> outcomes;
RWStructuredBuffer<float> masses;
uint targets;

float intersect(float3 origin, float3 direction, float reach)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.000001;
    ray.TMax = reach;
    RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
    q.TraceRayInline(scene, RAY_FLAG_NONE, 1, ray);
    while (q.Proceed()) {}
    return q.CommittedStatus() == COMMITTED_TRIANGLE_HIT ? q.CommittedRayT() : -1;
}

int readByte(int row)
{
    float hit = intersect(float3(-1, row, 0), float3(1, 0, 0), 257);
    return hit < 0 ? -1 : int(round(hit)) - 1;
}

int4 search(int t, int lower, out int lastByte)
{
    int history[8];
    for (int k = 0; k < 8; ++k) history[k] = readByte(t - 1 - k);
    lastByte = history[0];
    int calls = 8;
    int cursor = t - 1;
    int best = 0, prediction = -1, recent = -1;
    for (int i = 0; i < 16 && cursor > lower; ++i)
    {
        float distance = intersect(float3(history[0], cursor - 0.5, 0),
                                   float3(0, -1, 0), cursor - lower - 0.25);
        calls++;
        if (distance < 0) break;
        int row = int(round(cursor - 0.5 - distance));
        int length = 1;
        while (length < 8 && row - length >= lower)
        {
            int older = readByte(row - length);
            calls++;
            if (older != history[length]) break;
            length++;
        }
        if (length > best)
        {
            prediction = readByte(row + 1);
            calls++;
            best = length;
            if (i == 0) recent = prediction;
        }
        if (best == 8) break;
        cursor = row;
    }
    return int4(recent, prediction, best, calls);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void trace(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= targets) return;
    int t = int(tid.x) + 8;
    int lastByte;
    outcomes[tid.x] = search(t, max(0, t - 256), lastByte);
}

uint ringCursor;
uint ringCount;

[shader("compute")]
[numthreads(1, 1, 1)]
void live()
{
    int t = 256 + int(ringCursor);
    if (ringCount < 8)
    {
        outcomes[0] = int4(-1, -1, 0, readByte(t - 1));
        return;
    }
    int lastByte;
    int4 hit = search(t, t - int(ringCount), lastByte);
    outcomes[0] = int4(hit.xyz, lastByte);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void readMass(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= 65536) return;
    masses[tid.x] = intersect(float3(tid.x % 256, tid.x / 256, 0),
                               float3(0, 0, 1), 2);
}

RWStructuredBuffer<int> suffix;
[shader("compute")]
[numthreads(1, 1, 1)]
void recentBytes(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x < targets)
        suffix[tid.x] = readByte(int(256 + ringCursor - targets + tid.x));
}
