#include "episodic.slang"

RWStructuredBuffer<int> paths;
StructuredBuffer<uint> observed;
RWStructuredBuffer<uint> copied;

// Output layout: match length, number of tied paths, then up to 16 successors.
[shader("compute")]
[numthreads(64, 1, 1)]
void collect(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= targets) return;
    int t = int(tid.x) + 8;
    int low = max(0, t - 2048);
    int history[8];
    for (int j = 0; j < 8; ++j) history[j] = readByte(t - 1 - j, true);
    int column = 256 * history[1] + history[0];
    int cursor = t - 1, longest = 0, count = 0;
    int base = int(tid.x) * 18;
    for (int i = 0; i < 16 && cursor > low + 1; ++i)
    {
        float d = intersect(float3(column, cursor - 0.5, 0), float3(0, -1, 0), cursor - low - 1.25);
        if (d < 0) break;
        int end = int(round(cursor - 0.5 - d));
        int len = 2;
        while (len < 8 && end - len >= low && readByte(end - len, true) == history[len]) ++len;
        if (len > longest) { longest = len; count = 0; }
        if (len == longest) paths[base + 2 + count++] = readByte(end + 1, true);
        cursor = end;
    }
    paths[base] = longest;
    paths[base + 1] = count;
    for (int i = count; i < 16; ++i) paths[base + 2 + i] = -1;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void collectMass(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= targets) return;
    uint base = tid.x * 18;
    uint count = uint(paths[base + 1]);
    uint matches = 0;
    for (uint i = 0; i < count; ++i)
        if (paths[base + 2 + i] == int(observed[tid.x])) matches++;
    copied[tid.x] = matches;
}
