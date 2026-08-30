#pragma once

#include "l2_perception/inference/inference_backend.hpp"

#include <memory>

namespace L2Perception
{

// TensorRT 的公开边界与 OpenVINO 完全一致：L2 只看到 U8 NHWC 输入和
// float32 输出，不需要知道 CUDA buffer、execution context 或 TensorRT 的
// I/O tensor API。model_path 可以是 TensorRT 序列化 engine（.engine/.plan）
// 或 ONNX 文件；ONNX 会在 load() 阶段构建成 engine。
class TensorRtBackend final : public IInferenceBackend
{
public:
  TensorRtBackend();
  ~TensorRtBackend() override;

  TensorRtBackend(TensorRtBackend&&) noexcept;
  TensorRtBackend& operator=(TensorRtBackend&&) noexcept;
  TensorRtBackend(const TensorRtBackend&) = delete;
  TensorRtBackend& operator=(const TensorRtBackend&) = delete;

  void load(const InferenceModelConfig& config) override;
  bool ready() const noexcept override;
  const InferenceInputSpec& inputSpec() const override;
  [[nodiscard]] InferenceResult infer(const InferenceInput& input) override;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  InferenceInputSpec input_spec_;
  bool ready_{false};
};

}  // namespace L2Perception
