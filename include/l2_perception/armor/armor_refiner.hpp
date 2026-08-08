#pragma once

#include "l2_perception/armor.hpp"

#include <cstddef>
#include <vector>

#include <opencv2/core.hpp>

namespace L2Perception
{

// 单块装甲板的判定结果。三者必须严格区分：Rejected 表示"确认只有一根灯条"，
// NetworkKept 表示"检查本身没能给出结论"。把后者当成前者处理，会在远距离、
// 暗光或阈值偏高时成片误杀本来有效的检出。
enum class RefineVerdict
{
  Refined,      // 找到左右两根灯条，角点已被替换
  NetworkKept,  // 检查未生效或结果不可信，保留网络角点
  Rejected      // 确认单灯条遮挡，该检出应被丢弃
};

struct ArmorRefinerConfig
{
  bool enable{true};

  // ROI 相对网络角点包围盒的外扩比例。网络回归的角点常内缩在灯条内侧，
  // 外扩不足会把灯条头尾切掉，导致端点系统性偏短。
  float roi_expand_ratio{0.20F};

  // 以下二值化与灯条几何门限全部复刻 SP-Vision Detector 的口径，默认值取自
  // sp_vision_25-main/configs/standard3.yaml，便于两边直接对照调参。

  // 灰度二值化阈值（SP threshold）。刻意不用"ROI 最大值的比例"这类自适应阈值：
  // 实测相对阈值会把过曝灯条周围的光晕一并圈进来，minAreaRect 因此系统性偏长。
  double binary_threshold{150.0};

  // 灯条投影长度低于该像素数时（SP min_lightbar_length），传统检测找到一根还是
  // 两根纯粹取决于阈值抖动，结论没有意义，直接跳过检查并保留网络角点。
  // 这条是强制的：缺了它会把远距离目标成片丢掉，而那恰恰最需要维持跟踪。
  float min_lightbar_length_px{8.0F};

  // 灯条主轴与竖直方向的最大夹角（SP max_angle_error），单位 degree。
  float max_angle_error_deg{45.0F};

  // 灯条自身的长宽比门限（SP min/max_lightbar_ratio），排除反光点和大块高光。
  float min_lightbar_ratio{1.5F};
  float max_lightbar_ratio{20.0F};

  // 连通域中心到期望灯条中心的最大距离，单位是两根灯条间距的比例。
  // 超出则认为该连通域不属于这块装甲板（例如相邻装甲板渗入 ROI）。
  float max_assign_ratio{0.45F};

  // 连通域到左右两条中轴的距离之差小于该比例时，判定为过曝把两根灯条粘连成
  // 一块，属于检查失效而非遮挡证据。只数连通域个数会把这种情况误判成单灯条。
  float merged_blob_ratio{0.25F};

  // 只在投影宽高比低于该值（即明显侧对）时才允许判定单灯条遮挡。
  // 小板标称比 135:56≈2.41，取 1.2 相当于仅在偏转约 60° 以上时启用；
  // 大板标称比 230:56≈4.11，同一阈值相当于约 73°，偏保守。
  // 不按板型归一化是为了避免 L2 反向依赖 L3 的 ArmorType。
  float edge_on_aspect_ratio{1.2F};

  // 精修角点相对网络角点的最大位移，单位是装甲板对角线的比例。超出说明网络和
  // 传统两条通路互相矛盾，此时两者都不可信，保留网络角点而不是二选一。
  float max_corner_shift_ratio{0.25F};
};

// 一次 refine() 调用的统计。这个滤波是静默删除，不暴露计数的话，
// 误杀有效检出时只会表现为"跟踪莫名其妙断"，无法定位。
struct RefineStats
{
  std::size_t refined{0};
  std::size_t network_kept{0};
  std::size_t rejected{0};
};

// 单块装甲板的判定明细，供离线回放叠加显示。Rejected 的检出会被 refine() 从
// 结果里删掉，只有这条记录还留着它的 ROI 和角点，否则画面上根本无法标出
// "被拒的是哪一块"——而调阈值时唯一可靠的手段就是肉眼确认被划掉的确实该划。
struct RefineRecord
{
  cv::Rect roi;
  RefineVerdict verdict{RefineVerdict::NetworkKept};
  // 判定时的网络原始角点，以及精修后的角点（未精修时两者相同）。
  std::array<cv::Point2f, 4> network_corners{};
  std::array<cv::Point2f, 4> corners{};

  // 检查过程量，用于解释这块板为什么落到该判定。
  bool left_found{false};
  bool right_found{false};
  bool merged_blob{false};
  bool size_skipped{false};   // 灯条太短，检查被跳过
  bool shift_rejected{false};  // 精修位移过大，回退到网络角点
  float aspect_ratio{0.0F};    // 灯条间距 / 灯条长度，越小越侧对
  float lightbar_length{0.0F};
  float corner_shift{0.0F};
};

// 在网络检出的 ROI 内做传统灯条检测，用于两件事：把角点替换为灯条端点（精修），
// 以及在确认只有一根灯条时丢弃该检出（筛选）。
//
// 灯条提取与几何门限复刻 SP-Vision 的 Detector：灰度图 + 固定阈值 + minAreaRect，
// Lightbar 的 top/bottom/length/width/angle_error 定义与其逐项一致。区别只在于
// 这里的搜索范围被网络 ROI 限定，并且左右灯条由网络角点直接指派，因此不需要
// SP-Vision 的全图两两配对和共用灯条去重。
//
// 注意精修无法修复单灯条工况：图像里不存在第二根灯条时，网络给出的那两个角点
// 是凭空生成的，任何像素级处理都恢复不了不存在的信息。此时唯一正确的动作是拒绝。
class ArmorRefiner
{
public:
  explicit ArmorRefiner(ArmorRefinerConfig config = {});

  // 就地精修 armors 并剔除单灯条检出。返回本次调用的统计供上层观测。
  // records 非空时额外写出每块板的判定明细（含被拒者），供离线回放叠加显示；
  // 实机路径传 nullptr 即可，不会产生额外分配。
  RefineStats refine(const cv::Mat& image, std::vector<Armor>& armors,
                     std::vector<RefineRecord>* records = nullptr) const;

  [[nodiscard]] const ArmorRefinerConfig& config() const noexcept { return config_; }

private:
  [[nodiscard]] RefineVerdict refineOne(const cv::Mat& image, Armor& armor,
                                        RefineRecord* record) const;

  ArmorRefinerConfig config_;
};

}  // namespace L2Perception
