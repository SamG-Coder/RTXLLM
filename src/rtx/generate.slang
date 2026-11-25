#include "reflect_to_receiver.slangh"

// One group owns one generated sequence. Draws are input-independent seeded
// uniform thresholds. Each next context depends on the previous traced votes.
StructuredBuffer<uint> draws;
RWStructuredBuffer<int> sampled;
uint count;
int startByte;
groupshared uint histogram[257];
groupshared int currentByte;

[shader("compute")]
[numthreads(32, 1, 1)]
void trace(uint lane : SV_GroupIndex)
{
    if (lane == 0)
        currentByte = startByte;
    GroupMemoryBarrierWithGroupSync();
    for (uint step = 0; step < count; ++step)
    {
        for (uint i = lane; i < 257; i += 32)
            histogram[i] = 0;
        GroupMemoryBarrierWithGroupSync();

        int bank = reflectToReceiver(256 + currentByte, 32);
        int receiver = bank >= 512 && bank < 520 ? reflectToReceiver(bank, int(lane)) : -1;
        uint bin = receiver >= 0 && receiver < 256 ? uint(receiver) : 256;
        InterlockedAdd(histogram[bin], 1);
        GroupMemoryBarrierWithGroupSync();

        if (lane == 0)
        {
            currentByte = -1;
            if (histogram[256] == 0)
            {
                uint cumulative = 0;
                for (uint token = 0; token < 256; ++token)
                {
                    // 0.99 * votes/32 + 0.01/256, in units of 1/25600.
                    cumulative += 792 * histogram[token] + 1;
                    if (draws[step] < cumulative)
                    {
                        currentByte = int(token);
                        break;
                    }
                }
            }
            sampled[step] = currentByte;
        }
        GroupMemoryBarrierWithGroupSync();
        if (currentByte < 0)
            return;
    }
}
