#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace L2Perception
{

// 后端中立的 float32 输出张量。
// 例如形状 {1, 14, 8400} 只说明三个维度的大小；第 0 行是不是 x 坐标，
// 必须由 ArmorDecoder（或 BuffDecoder）根据实际模型约定解释。
class InferenceTensor
{
public:
  // OpenVINO / TensorRT 模型输出节点的名字，例如 "output0"。
  std::string name;

  // 张量形状，按模型原始维度保存，不在这里强制转置。
  std::vector<std::size_t> shape;

  // Mock/CPU 后端等需要自有输出时使用 vector。
  void setOwnedData(std::vector<float> data);

  // 推理输出可引用后端缓冲区而不复制。owner 必须保证数据有效且不会被下一帧覆写。
  void setExternalView(
    const float* data,
    std::size_t size,
    std::shared_ptr<const void> owner);

  // Decoder 统一通过这个只读 span 访问数据，不关心数据来自 vector 还是后端零拷贝视图。
  [[nodiscard]] std::span<const float> values() const noexcept;

  // shape 各维相乘得到理论元素数，例如 {1, 3, 640, 640} -> 1,228,800。
  [[nodiscard]] std::size_t elementCount() const noexcept;
  // 用于在 Decoder 前发现“shape 与实际数据长度不匹配”的后端错误。
  [[nodiscard]] bool isConsistent() const noexcept;

private:
  std::vector<float> owned_data_;
  const float* external_data_{nullptr};
  std::size_t external_size_{0};

  // 类型擦除的所有者。OpenVINO 后端放入 InferRequest 结果租约，公共头文件不依赖 SDK 类型。
  std::shared_ptr<const void> external_owner_;
};

// 一次推理可拥有多个输出张量，例如检测头、分割头或姿态头。
struct InferenceResult
{
  std::vector<InferenceTensor> outputs;

  // Decoder 优先按名字找输出，避免依赖“第 0 个输出”的不稳定顺序。
  [[nodiscard]] const InferenceTensor* findOutput(std::string_view name) const noexcept;
};

}  // namespace L2Perception
