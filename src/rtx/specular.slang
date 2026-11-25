#include "reflect_to_receiver.slangh"

StructuredBuffer<int> starts;
StructuredBuffer<int> programs;
RWStructuredBuffer<int> receivers;
uint queryCount;
uint programWidth;
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
        station = reflectToReceiver(station, relation);
        if (station < 0)
            break;
        receivers[row + hop] = station;
    }
}
