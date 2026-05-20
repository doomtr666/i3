// Helper pass: copies a u32 counter to the x-component of a VkDispatchIndirectCommand.
// Single thread dispatch (1, 1, 1). Used twice per frame:
//   - dirty_count → dispatch_dirty.x
//   - job_count   → dispatch_jobs.x

[[vk::binding(0, 0)]] RWByteAddressBuffer src_count;
[[vk::binding(1, 0)]] RWByteAddressBuffer dst_dispatch;

[shader("compute")]
[numthreads(1, 1, 1)]
void csMain()
{
    uint count = src_count.Load(0);
    // VkDispatchIndirectCommand: {x, y, z} — y and z were already set to 1 by reset pass
    dst_dispatch.Store(0, count);
}
