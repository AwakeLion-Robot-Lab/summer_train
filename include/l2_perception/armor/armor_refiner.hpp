#pragma once

#include "l2_perception/armor.hpp"

#include <array>
#include <cstddef>
#include <vector>

#include <opencv2/core.hpp>

namespace L2Perception
{

// 灯条精修只有 Refined / NetworkKept 两种结果：矫正失败时保留网络角点，绝不
// 丢掉检出。Rejected 只为离线统计接口保留，本实现不会产生。
enum class RefineVerdict
{
  Refined,      // 找到左右两根灯条，角点已被替换
  NetworkKept,  // 传统矫正失败，原样保留网络识别结果
  Rejected      // 兼容旧统计字段；本实现不会返回此值
};

struct ArmorRefinerConfig
{
  bool enable{true};

  // 传统灯条识别参数。
  double binary_threshold{150.0};
  float min_lightbar_length_px{8.0F};
  float max_angle_error_deg{45.0F};
  float min_lightbar_ratio{1.5F};
  float max_lightbar_ratio{20.0F};

  // 左右灯条端点到网络角点的距离和必须小于该值，传统角点才会覆盖网络角点。
  float max_endpoint_distance_px{15.0F};
};

// 一次 refine() 调用的统计。矫正不会删除网络结果，因此 rejected 恒为 0。
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

// 在网络检出的 ROI 内跑传统灯条检测，对单个 Armor 做二次角点矫正：灯条端点比
// 网络角点稳定，PnP 对角点误差又极敏感。找不到可信的两根灯条时返回 false，并
// 完整保留网络识别结果。
class ArmorRefiner
{
public:
  explicit ArmorRefiner(ArmorRefinerConfig config = {});

  // 单目标接口。成功时就地替换 armor.corners 并返回 true；失败时 armor 不变。
  bool detect(Armor& armor, const cv::Mat& bgr_img) const;

  // 批量接口，内部逐个调用上面的单目标接口。矫正不会删除网络检出，因此
  // rejected 始终为 0，失败项计入 network_kept。
  RefineStats refine(const cv::Mat& image, std::vector<Armor>& armors,
                     std::vector<RefineRecord>* records = nullptr) const;

  [[nodiscard]] const ArmorRefinerConfig& config() const noexcept { return config_; }

private:
  bool detectOne(Armor& armor, const cv::Mat& bgr_img, RefineRecord* record) const;

  ArmorRefinerConfig config_;
};

}  // namespace L2Perception
