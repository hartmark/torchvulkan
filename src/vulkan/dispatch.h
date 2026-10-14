#pragma once
#include <volk.h>
#include <memory>
#include <vector>

struct DeviceContext;
struct OpInfo;

class DAGDispatcher
{
public:
    explicit DAGDispatcher(DeviceContext* device) : device(device) {}
    void dispatch(const std::vector<std::unique_ptr<OpInfo>>& operations);

private:
    DeviceContext* device = nullptr;

    void recordOp(VkCommandBuffer cmd, const OpInfo* op);
    void recordCompute(VkCommandBuffer cmd, const OpInfo* op);
    void recordCopy(VkCommandBuffer cmd, const OpInfo* op);
    void recordBarrier(VkCommandBuffer cmd);
    void recordHostBarrier(VkCommandBuffer cmd);
};
