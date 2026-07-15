#include "l2_perception/inference/inference_backend.hpp"

#include <numeric>
#include <stdexcept>
#include <utility>

namespace L2Perception
{

void InferenceInput::setOwnedData(std::vector<std::uint8_t> data)
{
  owned_data_ = std::move(data);
}

std::span<const std::uint8_t> InferenceInput::values() const noexcept
{
  return owned_data_;
}

std::size_t InferenceInput::elementCount() const noexcept
{
  if (shape.empty()) {
    return 0;
  }

  return std::accumulate(
    shape.begin(), shape.end(), std::size_t{1},
    [](std::size_t total, std::size_t dimension) {
      return total * dimension;
    });
}

bool InferenceInput::isConsistent() const noexcept
{
  // U8 每个元素正好一个字节，所以元素数必须与 vector 长度一致。
  return elementCount() == owned_data_.size();
}

void InferenceTensor::setOwnedData(std::vector<float> data)
{
  owned_data_ = std::move(data);
  external_data_ = nullptr;
  external_size_ = 0;
  external_owner_.reset();
}

void InferenceTensor::setExternalView(
  const float* data,
  std::size_t size,
  std::shared_ptr<const void> owner)
{
  if ((data == nullptr && size != 0) || !owner) {
    throw std::invalid_argument("InferenceTensor external view requires valid data and owner");
  }

  owned_data_.clear();
  external_data_ = data;
  external_size_ = size;
  external_owner_ = std::move(owner);
}

std::span<const float> InferenceTensor::values() const noexcept
{
  if (external_owner_) {
    return {external_data_, external_size_};
  }
  return owned_data_;
}

std::size_t InferenceTensor::elementCount() const noexcept
{
  // 空 shape 不代表标量模型输出；当前架构不支持该情况，返回 0 让 isConsistent() 失败。
  if (shape.empty()) {
    return 0;
  }

  return std::accumulate(
    shape.begin(), shape.end(), std::size_t{1},
    [](std::size_t total, std::size_t dimension) {
      return total * dimension;
    });
}

bool InferenceTensor::isConsistent() const noexcept
{
  // 这里不解释 shape 的维度含义，只检查“维度乘积 == 实际 float 数”。
  return elementCount() == values().size();
}

const InferenceTensor* InferenceResult::findOutput(std::string_view name) const noexcept
{
  // 返回内部指针只用于当前 result 的只读访问；result 销毁后该指针不能保存。
  for (const auto& output : outputs) {
    if (output.name == name) {
      return &output;
    }
  }

  return nullptr;
}

}  // namespace L2Perception
