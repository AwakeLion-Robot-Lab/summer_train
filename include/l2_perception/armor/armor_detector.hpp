#pragma once

#include "l2_perception/armor/armor_decoder.hpp"
#include "l2_perception/armor/armor_refiner.hpp"
#include "l2_perception/inference/inference_backend.hpp"
#include "l2_perception/inference/image_preprocessor.hpp"

#include <memory>
#include <vector>

#include <opencv2/core.hpp>

namespace L2Perception
{

// L2 装甲检测编排层：图像预处理 → 原始推理 → 装甲模型解码 → 灯条精修与筛选。
// 它不拥有 PnP、跟踪或开火策略；这些工作在 L3/L4/L5。
class ArmorDetector
{
public:
  // 默认构造表示“未配置模型”，detect() 会安全返回空结果，便于 runtime 先启动相机和串口。
  ArmorDetector() = default;
  ArmorDetector(
    std::unique_ptr<IInferenceBackend> backend,
    ArmorDecoderConfig decoder_config = {},
    ImagePreprocessConfig preprocess_config = {},
    ArmorRefinerConfig refiner_config = {});

  [[nodiscard]] bool ready() const noexcept;
  // 一帧同步检测。Backend/Decoder 抛出的异常会被转换为日志和空结果，避免中断主循环。
  [[nodiscard]] std::vector<Armor> detect(const cv::Mat& image) const;

  // 最近一次 detect() 的精修/筛选统计。精修筛选会静默删除检出，
  // 不把计数暴露出来就无法判断它是否在误杀有效目标。
  [[nodiscard]] const RefineStats& lastRefineStats() const noexcept { return last_refine_stats_; }

  // 逐块判定明细，含已被删除的 Rejected 检出。仅在 collectRefineRecords(true)
  // 之后才填充，实机路径保持关闭以免每帧多一次分配。
  [[nodiscard]] const std::vector<RefineRecord>& lastRefineRecords() const noexcept
  {
    return last_refine_records_;
  }
  void collectRefineRecords(bool enable) noexcept { collect_refine_records_ = enable; }

private:
  std::unique_ptr<IInferenceBackend> backend_;
  ArmorDecoder decoder_{};
  ImagePreprocessConfig preprocess_config_{};
  ArmorRefiner refiner_{};
  bool collect_refine_records_{false};
  // detect() 对外是 const 的只读操作，统计只作为调试快照，与 Tracker::observations() 同理。
  mutable RefineStats last_refine_stats_{};
  mutable std::vector<RefineRecord> last_refine_records_{};
};

}  // namespace L2Perception
