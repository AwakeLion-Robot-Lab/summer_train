#pragma once

#include "l2_perception/inference/inference_backend.hpp"

#include <memory>

namespace L2Perception
{

/**
 * @brief OpenVINO 推理后端
 *
 * OpenVINO 接收通用 U8 NHWC BGR letterbox，在预处理图中完成颜色、归一化、
 * NHWC->NCHW 以及到原始模型 FP16/FP32 输入类型的转换。
 */
class OpenVinoBackend final : public IInferenceBackend
{
public:
  /**
   * @brief 构造一个未加载模型的后端
   */
  OpenVinoBackend();

  /**
   * @brief 析构；已借出的推理结果仍由租约保活
   */
  ~OpenVinoBackend() override;

  /**
   * @brief 移动构造
   */
  OpenVinoBackend(OpenVinoBackend&&) noexcept;

  /**
   * @brief 移动赋值
   * @return *this
   */
  OpenVinoBackend& operator=(OpenVinoBackend&&) noexcept;

  OpenVinoBackend(const OpenVinoBackend&) = delete;
  OpenVinoBackend& operator=(const OpenVinoBackend&) = delete;

  /**
   * @brief 读取模型、构建预处理图并编译，见 IInferenceBackend::load
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
   * @return 全部输出张量，零拷贝引用请求池里的缓冲区
   */
  [[nodiscard]] InferenceResult infer(const InferenceInput& input) override;

private:
  struct Impl;

  /// Impl 内的共享请求池可被结果租约保活，Backend 先析构时输出缓冲区仍然有效。
  std::shared_ptr<Impl> impl_;

  /// 宿主侧输入规格
  InferenceInputSpec input_spec_;

  /// 模型是否已加载
  bool ready_{false};
};

}  // namespace L2Perception
