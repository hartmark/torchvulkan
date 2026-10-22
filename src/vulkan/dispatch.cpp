#include "dispatch.h"
#include "device_context.h"
#include "shader.h"
#include <algorithm>
#include <unordered_map>

void DAGDispatcher::recordOp(VkCommandBuffer cmd, const OpInfo* op)
{
    switch (op->type)
    {
        case OpType::COMPUTE: recordCompute(cmd, op); break;
        case OpType::COPY: recordCopy(cmd, op); break;
    }
}

void DAGDispatcher::recordCompute(VkCommandBuffer cmd, const OpInfo* op)
{
    device->device_table.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, op->pipeline);
    if (op->pushConstantSize > 0)
    {
        device->device_table.vkCmdPushConstants(
            cmd, op->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
            0, op->pushConstantSize, op->pushConstantData.data()
        );
    }
    device->device_table.vkCmdDispatch(cmd, op->groupX, op->groupY, op->groupZ);
}

void DAGDispatcher::recordCopy(VkCommandBuffer cmd, const OpInfo* op)
{
    device->device_table.vkCmdCopyBuffer(cmd, op->srcBuffer, op->dstBuffer, 1, &op->region);
}

void DAGDispatcher::recordBarrier(VkCommandBuffer cmd)
{
    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
    barrier.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                            VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;

    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers    = &barrier;

    device->device_table.vkCmdPipelineBarrier2(cmd, &dep);
}

void DAGDispatcher::recordHostBarrier(VkCommandBuffer cmd)
{
    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
    barrier.dstStageMask  = VK_PIPELINE_STAGE_2_HOST_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;

    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers    = &barrier;

    device->device_table.vkCmdPipelineBarrier2(cmd, &dep);
}

void DAGDispatcher::dispatch(const std::vector<std::unique_ptr<OpInfo>>& operations)
{
    // all elements in the vector depend on the key
    std::unordered_map<const OpInfo*, std::vector<const OpInfo*>> dependents;
    std::unordered_map<const OpInfo*, int> unresolved_deps;
    std::unordered_map<uint64_t, const OpInfo*> last_writer;
    std::unordered_map<uint64_t, std::vector<const OpInfo*>> readers_since_write;

    for (const auto& owned : operations)
    {
        const OpInfo* op = owned.get();
        for (const MemoryRange& input : op->inputs)
        {
            // RAW: this read must wait for the previous write
            auto it = last_writer.find(input.base_addr);
            if (it != last_writer.end())
            {
                dependents[it->second].push_back(op);
                unresolved_deps[op]++;
            }
            readers_since_write[input.base_addr].push_back(op);
        }

        for (const MemoryRange& output : op->outputs)
        {
            // WAW: this write must wait for the previous writer
            auto it_write = last_writer.find(output.base_addr);
            if (it_write != last_writer.end() && it_write->second != op)
            {
                dependents[it_write->second].push_back(op);
                unresolved_deps[op]++;
            }

            // WAR: this write must wait for all readers since the last write
            auto it_read = readers_since_write.find(output.base_addr);
            if (it_read != readers_since_write.end())
            {
                for (const OpInfo* reader : it_read->second)
                {
                    if (reader == op) continue;
                    dependents[reader].push_back(op);
                    unresolved_deps[op]++;
                }
            }

            last_writer[output.base_addr] = op;
            readers_since_write.erase(output.base_addr);
        }
    }

    VkCommandBuffer cmd = device->cache.allocateCommandBuffer();

    std::vector<const OpInfo*> ready;
    for (const auto& op : operations)
    {
        if (unresolved_deps[op.get()] == 0) ready.push_back(op.get());
    }

    while (!ready.empty())
    {
        // we see an issue on NVIDIA where a vkCmdDispatch followed by a vkCmdCopyBuffer in the same wave lets the next wave read the 
        // dispatch's output before it was written. so we separate the copy ops from the dispatch ops with a barrier.
        auto first_dispatch = std::stable_partition(ready.begin(), ready.end(), [](const OpInfo* op) { return op->type == OpType::COPY; });
        for (auto it = ready.begin(); it != ready.end(); ++it)
        {
            if (it == first_dispatch && it != ready.begin()) recordBarrier(cmd);
            recordOp(cmd, *it);
        }

        std::vector<const OpInfo*> next_wave;
        for (const OpInfo* op : ready)
        {
            for (const OpInfo* dependent : dependents[op])
            {
                unresolved_deps[dependent]--;
                if (unresolved_deps[dependent] == 0) next_wave.push_back(dependent);
            }
        }

        if (!next_wave.empty()) recordBarrier(cmd);
        ready = std::move(next_wave);
    }

    recordHostBarrier(cmd);
    VK_CHECK(device->device_table.vkEndCommandBuffer(cmd));

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    VkFence fence = device->cache.allocateFence();
    VkResult submit_result = device->device_table.vkQueueSubmit(device->computeQueue, 1, &submitInfo, fence);
    if (submit_result != VK_SUCCESS) {
        device->cache.deleteCommandBuffer(cmd);
        device->cache.deleteFence(fence);
        TORCH_CHECK(false, "torchvulkan [ERROR]: vkQueueSubmit failed with VkResult ", submit_result, ".");
    }

    VkResult wait_result = device->device_table.vkWaitForFences(device->device, 1, &fence, VK_TRUE, UINT64_MAX);
    TORCH_CHECK(wait_result == VK_SUCCESS, "torchvulkan [ERROR]: Wait for fence failed with VkResult ", wait_result, ".");

    device->cache.deleteCommandBuffer(cmd);
    device->cache.deleteFence(fence);
}
