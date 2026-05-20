// Pass 0: reset counters and indirect dispatch args at the start of each frame.
// Single thread dispatch (1, 1, 1).

#include "brickmap_common.slang"

[[vk::binding(0, 0)]] RWByteAddressBuffer dirty_count;      // u32 counter
[[vk::binding(1, 0)]] RWByteAddressBuffer job_count;        // u32 counter
[[vk::binding(2, 0)]] RWByteAddressBuffer dispatch_dirty;   // VkDispatchIndirectCommand
[[vk::binding(3, 0)]] RWByteAddressBuffer dispatch_jobs;    // VkDispatchIndirectCommand

[shader("compute")]
[numthreads(1, 1, 1)]
void csMain()
{
    // Clear frame counters
    dirty_count.Store(0, 0u);
    job_count.Store(0, 0u);

    // Set y=1, z=1 in dispatch commands (x will be filled by indirect helper)
    // Layout: {x: u32, y: u32, z: u32}
    dispatch_dirty.Store(0,  0u);
    dispatch_dirty.Store(4,  1u);
    dispatch_dirty.Store(8,  1u);
    dispatch_jobs.Store(0,   0u);
    dispatch_jobs.Store(4,   1u);
    dispatch_jobs.Store(8,   1u);
}
