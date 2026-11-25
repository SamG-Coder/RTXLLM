// #91 (Claude): conventional comparator for E10 episodic search. No acceleration
// structure and no ray queries: the same search semantics as episodic.slang `search`
// (key width 1 or 2, at most 16 candidate endings, match length up to 8,
// nearest-first, stop at 8), implemented as a backward scan over the byte buffer.

StructuredBuffer<uint> bytes;      // one byte per element
RWStructuredBuffer<int4> outcomes; // (recent, prediction, best, candidates)
uint targets, windowSize, pairAddress;

[shader("compute")]
[numthreads(64, 1, 1)]
void scan(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= targets) return;
    int t = int(tid.x) + 8;
    int lower = max(0, t - int(windowSize));
    bool paired = pairAddress != 0;
    int width = paired ? 2 : 1;
    int history[8];
    for (int k = 0; k < 8; ++k) history[k] = int(bytes[t - 1 - k]);
    int best = 0, prediction = -1, recent = -1, found = 0;
    // A candidate ending e needs e - width + 1 >= lower and a successor e + 1 <= t - 1.
    for (int e = t - 2; e >= lower + width - 1 && found < 16; --e)
    {
        if (int(bytes[e]) != history[0]) continue;
        if (paired && int(bytes[e - 1]) != history[1]) continue;
        found++;
        int length = width;
        while (length < 8 && e - length >= lower && int(bytes[e - length]) == history[length])
            length++;
        if (length > best)
        {
            best = length;
            prediction = int(bytes[e + 1]);
            if (found == 1) recent = prediction;
        }
        if (best == 8) break;
    }
    outcomes[tid.x] = int4(recent, prediction, best, found);
}
