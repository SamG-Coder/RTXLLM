RaytracingAccelerationStructure scene;
StructuredBuffer<uint> starts, targetBytes;
RWStructuredBuffer<int2> counts;
RWStructuredBuffer<int> letters;
uint queries, delay, rowIndex;

int symbol(uint column, uint row)
{
    RayDesc ray;
    ray.Origin = float3(column, row, 0);
    ray.Direction = float3(0, 0, 1);
    ray.TMin = 0.01;
    ray.TMax = 258;
    RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
    q.TraceRayInline(scene, RAY_FLAG_NONE, 0xff, ray);
    while(q.Proceed()) {}
    return q.CommittedStatus() == COMMITTED_TRIANGLE_HIT ? int(round(q.CommittedRayT()))-1 : -1;
}

StructuredBuffer<uint> predecessors;
uint livePredecessor;

uint rowWeight(uint row, uint predecessor)
{
    if (predecessor == 0) return 1;
    int lo = symbol(0, row), hi = symbol(1, row);
    return (lo + 256 * hi == int(predecessor)) ? 8 : 1;
}

void compareRows(uint i, bool tagged)
{
    if (i >= queries) return;
    uint end = i > delay ? i-delay : 0;
    uint n = min(64u,end), same = 0, total = 0;
    uint skip = tagged ? 2 : 0;
    for (uint row=end-n; row<end; row++)
    {
        uint w = tagged ? rowWeight(row,predecessors[i]) : 1;
        total += w;
        uint k=0, length=starts[i+1]-starts[i];
        while (k<length && symbol(skip+k,row)==int(targetBytes[starts[i]+k])) k++;
        if (k==length && symbol(skip+k,row)==256) same += w;
    }
    counts[i]=int2(same,total);
}

[shader("compute")]
[numthreads(64,1,1)]
void score(uint3 tid:SV_DispatchThreadID) { compareRows(tid.x,false); }

[shader("compute")]
[numthreads(64,1,1)]
void scoreContext(uint3 tid:SV_DispatchThreadID) { compareRows(tid.x,true); }

[shader("compute")]
[numthreads(64,1,1)]
void liveWeights(uint3 tid:SV_DispatchThreadID)
{
    if (tid.x<queries) letters[tid.x]=rowWeight(tid.x,livePredecessor);
}

[shader("compute")]
[numthreads(64,1,1)]
void readRow(uint3 tid:SV_DispatchThreadID)
{
    if(tid.x<queries) letters[tid.x]=symbol(tid.x,rowIndex);
}
