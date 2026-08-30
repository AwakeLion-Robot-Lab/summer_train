#pragma once

#include "l2_perception/armor.hpp"

#include <array>
#include <cstddef>
#include <vector>

#include <opencv2/core.hpp>

namespace L2Perception
{

// SP 风格接口只有 Refined / NetworkKept 两种运行结果。Rejected 暂时保留，
// 避免现有离线回放和统计接口失去源码兼容性，但不会由本实现产生。
enum class RefineVerdict
{
  Refined,      // 找到左右两根灯条，角点已被替换
  NetworkKept,  // 传统矫正失败，原样保留网络识别结果
  Rejected      // 兼容旧统计字段；SP 风格接口不会返回此值
};

struct ArmorRefinerConfig
{
  bool enable{true};

  // 以下参数与 sp_vision_25-main/configs/standard3.yaml 的传统识别参数一致。
  double binary_threshold{150.0};
  float min_lightbar_length_px{8.0F};
  float max_angle_error_deg{45.0F};
  float min_lightbar_ratio{1.5F};
  float max_lightbar_ratio{20.0F};

  // SP Detector::detect(Armor&, image) 中写死为 15 px：左右灯条端点到网络角点的
  // 距离和必须小于该值，传统角点才会覆盖网络角点。
  float max_endpoint_distance_px{15.0F};
};

// 一次 refine() 调用的统计。SP 风格矫正不会删除网络结果，因此 rejected 恒为 0。
struct RefineStats
{
  std::size_t refined{0};
  std::size_t network_kept{0};
  std::size_t rejected{0};
};

// 单块装甲板的判定明细，供离线回放叠加显示。
struct RefineRecord
{
  cv::Rect roi;
  RefineVerdict verdict{RefineVerdict::NetworkKept};
  std::array<cv::Point2f, 4> network_corners{};
  std::array<cv::Point2f, 4> corners{};

  bool left_found{false};
  bool right_found{false};
  bool merged_blob{false};
  bool size_skipped{false};
  bool shift_rejected{false};
  float aspect_ratio{0.0F};
  float lightbar_length{0.0F};
  float corner_shift{0.0F};
};

// 在网络检出的 ROI 内复用 SP-Vision 的传统灯条检测，对单个 Armor 做二次角点矫正。
// 找不到可信的两根灯条时返回 false，并完整保留网络识别结果。
class ArmorRefiner
{
public:
  explicit ArmorRefiner(ArmorRefinerConfig config = {});

  // 与 SP-Vision Detector::detect(Armor&, const cv::Mat&) 相同的单目标接口。
  // 成功时就地替换 armor.corners 并返回 true；失败时 armor 保持不变。
  bool detect(Armor& armor, const cv::Mat& bgr_img) const;

  // 批量兼容接口，内部逐个调用上述 SP 风格接口。SP 的传统矫正不会删除网络检出，
  // 因此 rejected 始终为 0，失败项计入 network_kept。
  RefineStats refine(const cv::Mat& image, std::vector<Armor>& armors,
                     std::vector<RefineRecord>* records = nullptr) const;

  const ArmorRefinerConfig& config() const noexcept { return config_; }

private:
  bool detectOne(Armor& armor, const cv::Mat& bgr_img, RefineRecord* record) const;

  ArmorRefinerConfig config_;
};

}  // namespace L2Perception
