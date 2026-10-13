#pragma once

#include "l2_perception/inference/inference_backend.hpp"

#include <memory>

namespace L2Perception
{

/**
 * @brief TensorRT 推理后端
 *
 * TensorRT 的公开边界与 OpenVINO 完全一致：L2 只看到 U8 NHWC 输入和
 * float32 输出，不需要知道 CUDA buffer、execution context 或 TensorRT 的
 * I/O tensor API。model_path 可以是 TensorRT 序列化 engine（.engine/.plan）
 * 或 ONNX 文件；ONNX 会在 load() 阶段构建成 engine。
 */
class TensorRtBackend final : public IInferenceBackend
{
public:
  /**
   * @brief 构造一个未加载模型的后端
   */
  TensorRtBackend();

  /**
   * @brief 析构，释放 engine、context 与 CUDA 缓冲区
   */
  ~TensorRtBackend() override;

  /**
   * @brief 移动构造
   */
  TensorRtBackend(TensorRtBackend&&) noexcept;

  /**
   * @brief 移动赋值
   * @return *this
   */
  TensorRtBackend& operator=(TensorRtBackend&&) noexcept;

  TensorRtBackend(const TensorRtBackend&) = delete;
  TensorRtBackend& operator=(const TensorRtBackend&) = delete;

  /**
   * @brief 加载 engine，或把 ONNX 构建成 engine，见 IInferenceBackend::load
   * @param config 模型路径、设备和调度参数
   */
  void load(const InferenceModelConfig& config) override;

  /**
   * @brief 模型是否已加载
   * @return 已加载返回 true
   */
  bool ready() const noexcept override;

  /**
   * @brief 宿主侧输入的名字和形状
   * @return 输入规格
   */
  const InferenceInputSpec& inputSpec() const override;

  /**
   * @brief 同步推理，见 IInferenceBackend::infer
   * @param input U8 NHWC BGR 输入
   * @return 全部输出张量
   */
  [[nodiscard]] InferenceResult infer(const InferenceInput& input) override;

private:
  struct Impl;

  /// TensorRT 运行时对象和 CUDA 缓冲区
  std::unique_ptr<Impl> impl_;

  /// 宿主侧输入规格
  InferenceInputSpec input_spec_;

  /// 模型是否已加载
  bool ready_{false};
};

}  // namespace L2Perception
