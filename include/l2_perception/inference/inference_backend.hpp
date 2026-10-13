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

/**
 * @brief 模型训练时要求的输入颜色顺序
 */
enum class ModelColorOrder
{
  Bgr,
  Rgb
};

/**
 * @brief 推理后端种类
 */
enum class InferenceBackendKind
{
  OpenVino,
  TensorRt
};

/**
 * @brief 大小核调度类型，取值与 OpenVINO 的 ov::hint::SchedulingCoreType 一一对应
 *
 * 这里另立一份是为了不让公开头文件依赖 OpenVINO SDK——同一个理由，后端类
 * 用的是 PIMPL。
 */
enum class SchedulingCoreType
{
  Any,        ///< 交给插件决定，会同时使用 P 核和 E 核
  PCoreOnly,  ///< 只用性能核
  ECoreOnly   ///< 只用能效核
};

/**
 * @brief 仅包含所有后端都能理解的模型信息
 *
 * OpenVINO 的 device 可为 "CPU"、"GPU"；TensorRT 接受 "CUDA"、"CUDA:<index>"、
 * "GPU" 或 "GPU:<index>"，并将其解释为 CUDA device 选择。
 * 后端专属参数不要塞进这个公共结构体。
 */
struct InferenceModelConfig
{
  /// 模型文件路径
  std::filesystem::path model_path;

  /// 推理设备
  std::string device{"CPU"};

  /// 相机交给 Backend 的输入固定为 BGR；这里声明训练时模型需要的颜色顺序。
  ModelColorOrder model_color_order{ModelColorOrder::Rgb};

  /// 255 表示把像素除以 255，得到 [0, 1]。这是除数，不是原来的 1/255 乘数。
  float normalization_divisor{255.0F};

  /// 自瞄是单帧同步链路，要的是单次延迟而不是吞吐，因此固定用 LATENCY 提示。
  bool latency_hint{true};

  /// 0 表示交给 OpenVINO 自行决定。在大小核 CPU 上把线程摊到 E 核会让 P 核在每个
  /// 同步点空等，实测反而更慢，所以宁可显式限制线程数，也不要默认铺满所有核。
  std::size_t inference_num_threads{0};

  /// 大小核调度。Any 保持 OpenVINO 默认（会同时用 P 核和 E 核）。
  ///
  /// 自瞄是单帧同步链路：一次推理内部按算子并行，每层结束都有一个同步点。
  /// P 核和 E 核混用时，整层的耗时由最慢的那份切分决定，也就是被 E 核拖住，
  /// 而 P 核在同步点上白等。所以延迟敏感的场景通常 PCoreOnly 更快，哪怕
  /// 总算力更小。ECoreOnly 留给"把 P 核让给别的进程"这种取舍。
  SchedulingCoreType scheduling_core_type{SchedulingCoreType::Any};

  /// 超线程。nullopt 保持 OpenVINO 默认。同一个物理 P 核上的两个逻辑线程共享
  /// 执行单元和 L1/L2，算子级并行下互相抢资源，单次延迟往往不降反升；关掉它
  /// 通常和 PCoreOnly 一起用。吞吐场景则相反，所以这里不给硬编码默认值。
  std::optional<bool> enable_hyper_threading{};
};

/**
 * @brief 模型输入节点的名字和宿主侧形状
 *
 * 所有后端对宿主侧输入使用同一契约：uint8、NHWC、BGR，例如 {1, 640, 640, 3}。
 * OpenVINO 在预处理图内转换；TensorRT 在后端把相同的颜色、归一化和布局转换
 * 写入 CUDA 输入 buffer。
 */
struct InferenceInputSpec
{
  /// 输入节点名
  std::string name;

  /// 宿主侧形状，NHWC
  std::vector<std::size_t> shape;
};

/**
 * @brief 一帧预处理后的宿主输入
 *
 * 它只拥有连续 U8 NHWC 数据，不承担模型输出职责。
 */
class InferenceInput
{
public:
  /// 输入节点名
  std::string name;

  /// 形状，NHWC
  std::vector<std::size_t> shape;

  /**
   * @brief 接管一块连续的 U8 NHWC 数据
   * @param data 像素数据
   */
  void setOwnedData(std::vector<std::uint8_t> data);

  /**
   * @brief 只读访问像素数据
   * @return 数据视图
   */
  std::span<const std::uint8_t> values() const noexcept;

  /**
   * @brief shape 各维相乘得到的理论元素数
   * @return 元素数
   */
  std::size_t elementCount() const noexcept;

  /**
   * @brief shape 与实际数据长度是否一致
   * @return 一致返回 true
   */
  bool isConsistent() const noexcept;

private:
  /// 自有的像素数据
  std::vector<std::uint8_t> owned_data_;
};

/**
 * @brief 推理后端接口
 */
class IInferenceBackend
{
public:
  virtual ~IInferenceBackend() = default;

  /**
   * @brief 读取模型并创建可执行上下文，启动阶段调用
   *
   * 失败时抛出带原因的异常。一般在创建 ArmorDetector 前完成，不要在每帧
   * detect() 中重复 load()。
   *
   * @param config 模型路径、设备和调度参数
   */
  virtual void load(const InferenceModelConfig& config) = 0;

  /**
   * @brief 模型是否已加载
   * @return 已加载返回 true
   */
  virtual bool ready() const noexcept = 0;

  /**
   * @brief 宿主侧输入的名字和形状
   * @return 输入规格
   */
  virtual const InferenceInputSpec& inputSpec() const = 0;

  /**
   * @brief 同步推理
   *
   * 返回结果可以拥有数据，也可以用带生命周期租约的只读零拷贝视图。
   *
   * @param input U8 NHWC 输入，input.values() 在 infer() 返回前必须有效
   * @return 全部输出张量
   */
  [[nodiscard]] virtual InferenceResult infer(const InferenceInput& input) = 0;
};

/**
 * @brief 用一帧全零输入跑一次推理，取回各输出节点的名字和形状
 *
 * 这是启动阶段用来发现模型契约的，每帧路径绝不要调用。放在推理层是因为
 * "怎么问出输出形状"是后端的事；"这个形状对应哪种装甲板字段布局"则是
 * L2 装甲模块的事，两者不要混在一起。
 *
 * @param backend 已加载模型的后端
 * @return 各输出节点的名字和形状
 */
[[nodiscard]] std::vector<InferenceOutputSpec> probeOutputSpecs(IInferenceBackend& backend);

/**
 * @brief 后端种类的名字
 * @param backend 后端种类
 * @return "openvino" 或 "tensorrt"
 */
std::string_view backendName(InferenceBackendKind backend) noexcept;

/**
 * @brief 解析后端名字，忽略大小写和非字母数字字符
 * @param name 后端名，接受 openvino、tensorrt、trt
 * @return 后端种类；无法识别时为空
 */
[[nodiscard]] std::optional<InferenceBackendKind> parseBackend(
  std::string_view name);

/**
 * @brief 创建一个未加载模型的后端
 * @param backend 后端种类
 * @return 新建的后端
 */
[[nodiscard]] std::unique_ptr<IInferenceBackend> makeBackend(
  InferenceBackendKind backend);

}  // namespace L2Perception
