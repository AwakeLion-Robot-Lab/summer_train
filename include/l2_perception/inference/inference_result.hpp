#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace L2Perception
{

/**
 * @brief 后端中立的 float32 输出张量
 *
 * 例如形状 {1, 14, 8400} 只说明三个维度的大小；第 0 行是不是 x 坐标，
 * 必须由 ArmorDecoder（或 BuffDecoder）根据实际模型约定解释。
 */
class InferenceTensor
{
public:
  /// OpenVINO / TensorRT 模型输出节点的名字，例如 "output0"。
  std::string name;

  /// 张量形状，按模型原始维度保存，不在这里强制转置。
  std::vector<std::size_t> shape;

  /**
   * @brief 接管一块自有输出，Mock/CPU 后端等需要自有输出时使用
   * @param data 输出数据
   */
  void setOwnedData(std::vector<float> data);

  /**
   * @brief 引用后端缓冲区而不复制
   * @param data 缓冲区首地址
   * @param size 元素数
   * @param owner 保活缓冲区的所有者，必须保证数据有效且不会被下一帧覆写
   */
  void setExternalView(
    const float* data,
    std::size_t size,
    std::shared_ptr<const void> owner);

  /**
   * @brief 只读访问输出数据
   *
   * Decoder 统一通过这个 span 访问，不关心数据来自 vector 还是后端零拷贝视图。
   *
   * @return 数据视图
   */
  std::span<const float> values() const noexcept;

  /**
   * @brief shape 各维相乘得到的理论元素数，例如 {1, 3, 640, 640} -> 1,228,800
   * @return 元素数
   */
  std::size_t elementCount() const noexcept;

  /**
   * @brief 用于在 Decoder 前发现"shape 与实际数据长度不匹配"的后端错误
   * @return 一致返回 true
   */
  bool isConsistent() const noexcept;

private:
  /// 自有数据，使用外部视图时为空
  std::vector<float> owned_data_;

  /// 外部视图首地址
  const float* external_data_{nullptr};

  /// 外部视图元素数
  std::size_t external_size_{0};

  /// 类型擦除的所有者。OpenVINO 后端放入 InferRequest 结果租约，公共头文件不依赖 SDK 类型。
  std::shared_ptr<const void> external_owner_;
};

/**
 * @brief 模型输出节点的名字和形状，不含数据
 *
 * 用于在不做完整推理解码的前提下描述一个输出——例如启动阶段探测模型契约。
 * 与 InferenceInputSpec 对称，但放在这里而不是 backend 头里：解码侧要用它，
 * 却不该知道后端的存在。
 */
struct InferenceOutputSpec
{
  /// 输出节点名
  std::string name;

  /// 形状
  std::vector<std::size_t> shape;
};

/**
 * @brief 一次推理的全部输出
 *
 * 一次推理可拥有多个输出张量，例如检测头、分割头或姿态头。
 */
struct InferenceResult
{
  /// 输出张量
  std::vector<InferenceTensor> outputs;

  /**
   * @brief 按名字找输出
   *
   * Decoder 优先按名字找输出，避免依赖"第 0 个输出"的不稳定顺序。
   *
   * @param name 输出节点名
   * @return 找到的张量；不存在时为 nullptr
   */
  const InferenceTensor* findOutput(std::string_view name) const noexcept;
};

}  // namespace L2Perception
