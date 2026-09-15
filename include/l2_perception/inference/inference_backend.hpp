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

// 大小核调度类型，取值与 OpenVINO 的 ov::hint::SchedulingCoreType 一一对应。
// 这里另立一份是为了不让公开头文件依赖 OpenVINO SDK——同一个理由，后端类
// 用的是 PIMPL。
enum class SchedulingCoreType
{
  Any,        // 交给插件决定，会同时使用 P 核和 E 核
  PCoreOnly,  // 只用性能核
  ECoreOnly   // 只用能效核
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

  // 大小核调度。Any 保持 OpenVINO 默认（会同时用 P 核和 E 核）。
  //
  // 自瞄是单帧同步链路：一次推理内部按算子并行，每层结束都有一个同步点。
  // P 核和 E 核混用时，整层的耗时由最慢的那份切分决定，也就是被 E 核拖住，
  // 而 P 核在同步点上白等。所以延迟敏感的场景通常 PCoreOnly 更快，哪怕
  // 总算力更小。ECoreOnly 留给"把 P 核让给别的进程"这种取舍。
  SchedulingCoreType scheduling_core_type{SchedulingCoreType::Any};

  // 超线程。nullopt 保持 OpenVINO 默认。同一个物理 P 核上的两个逻辑线程共享
  // 执行单元和 L1/L2，算子级并行下互相抢资源，单次延迟往往不降反升；关掉它
  // 通常和 PCoreOnly 一起用。吞吐场景则相反，所以这里不给硬编码默认值。
  std::optional<bool> enable_hyper_threading{};
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
  std::span<const std::uint8_t> values() const noexcept;
  std::size_t elementCount() const noexcept;
  bool isConsistent() const noexcept;

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

  virtual bool ready() const noexcept = 0;
  virtual const InferenceInputSpec& inputSpec() const = 0;

  // 同步推理。U8 NHWC input.values() 在 infer() 返回前必须有效。
  // 返回结果可以拥有数据，也可以用带生命周期租约的只读零拷贝视图。
  [[nodiscard]] virtual InferenceResult infer(const InferenceInput& input) = 0;
};

// 用一帧全零输入跑一次推理，取回各输出节点的名字和形状。
// 这是启动阶段用来发现模型契约的，每帧路径绝不要调用。放在推理层是因为
// "怎么问出输出形状"是后端的事；"这个形状对应哪种装甲板字段布局"则是
// L2 装甲模块的事，两者不要混在一起。
[[nodiscard]] std::vector<InferenceOutputSpec> probeOutputSpecs(IInferenceBackend& backend);

std::string_view inferenceBackendName(InferenceBackendKind backend) noexcept;
[[nodiscard]] std::optional<InferenceBackendKind> inferenceBackendFromString(
  std::string_view name);
[[nodiscard]] std::unique_ptr<IInferenceBackend> makeInferenceBackend(
  InferenceBackendKind backend);

}  // namespace L2Perception
