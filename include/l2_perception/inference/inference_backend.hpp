#pragma once

#include "l2_perception/inference/inference_result.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace L2Perception
{

enum class ModelColorOrder
{
  Bgr,
  Rgb
};

enum class InferenceBackendKind
{
  OpenVino,
  TensorRt
};

// 仅包含所有后端都能理解的模型信息。
// OpenVINO 的 device 可为 "CPU"、"GPU"；TensorRT 接受 "CUDA"、"CUDA:<index>"、
// "GPU" 或 "GPU:<index>"，并将其解释为 CUDA device 选择。
// 后端专属参数不要塞进这个公共结构体。
struct InferenceModelConfig
{
  std::filesystem::path model_path;
  std::string device{"CPU"};

  // 相机交给 Backend 的输入固定为 BGR；这里声明训练时模型需要的颜色顺序。
  ModelColorOrder model_color_order{ModelColorOrder::Rgb};

  // 255 表示把像素除以 255，得到 [0, 1]。这是除数，不是原来的 1/255 乘数。
  float normalization_divisor{255.0F};

  // 自瞄是单帧同步链路，要的是单次延迟而不是吞吐，因此固定用 LATENCY 提示。
  bool latency_hint{true};

  // 0 表示交给 OpenVINO 自行决定。在大小核 CPU 上把线程摊到 E 核会让 P 核在每个
  // 同步点空等，实测反而更慢，所以宁可显式限制线程数，也不要默认铺满所有核。
  std::size_t inference_num_threads{0};

  // 可选的大小核调度提示。SP 只设置 LATENCY，不额外限制 P 核，因此默认关闭。
  bool prefer_performance_cores{false};
};

// 所有后端对宿主侧输入使用同一契约：uint8、NHWC、BGR，例如 {1, 640, 640, 3}。
// OpenVINO 在预处理图内转换；TensorRT 在后端把相同的颜色、归一化和布局转换
// 写入 CUDA 输入 buffer。
struct InferenceInputSpec
{
  std::string name;
  std::vector<std::size_t> shape;
};

// 一帧预处理后的宿主输入。它只拥有连续 U8 NHWC 数据，不承担模型输出职责。
class InferenceInput
{
public:
  std::string name;
  std::vector<std::size_t> shape;

  void setOwnedData(std::vector<std::uint8_t> data);
  [[nodiscard]] std::span<const std::uint8_t> values() const noexcept;
  [[nodiscard]] std::size_t elementCount() const noexcept;
  [[nodiscard]] bool isConsistent() const noexcept;

private:
  std::vector<std::uint8_t> owned_data_;
};

class IInferenceBackend
{
public:
  virtual ~IInferenceBackend() = default;

  // 启动阶段调用：读取模型并创建可执行上下文。失败时抛出带原因的异常。
  // 一般在创建 ArmorDetector 前完成，不要在每帧 detect() 中重复 load()。
  virtual void load(const InferenceModelConfig& config) = 0;

  [[nodiscard]] virtual bool ready() const noexcept = 0;
  [[nodiscard]] virtual const InferenceInputSpec& inputSpec() const = 0;

  // 同步推理。U8 NHWC input.values() 在 infer() 返回前必须有效。
  // 返回结果可以拥有数据，也可以用带生命周期租约的只读零拷贝视图。
  [[nodiscard]] virtual InferenceResult infer(const InferenceInput& input) = 0;
};

[[nodiscard]] std::string_view inferenceBackendName(InferenceBackendKind backend) noexcept;
[[nodiscard]] std::optional<InferenceBackendKind> inferenceBackendFromString(
  std::string_view name);
[[nodiscard]] std::unique_ptr<IInferenceBackend> makeInferenceBackend(
  InferenceBackendKind backend);

}  // namespace L2Perception
