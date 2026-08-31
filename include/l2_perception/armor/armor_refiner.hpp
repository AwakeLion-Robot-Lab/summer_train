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

  // 灯条端点改用亮度梯度 + PCA 主轴定位，而不是 minAreaRect 的二值边界。
  //
  // minAreaRect 的端点由阈值决定：阈值偏低斑点膨胀、偏高收缩，误差方向正好
  // 沿灯条轴向——那是 UVL 观测里灯条长度与左右间距两个深度线索最怕的方向。
  // PCA 版只用阈值圈 ROI，端点由灰度突变找，对曝光不敏感。
  //
  // 在几何筛选之后、配对之前执行，所以 max_endpoint_distance_px 那道门限
  // 看到的也是修正后的端点。
  bool pca_corner_correction{true};

  // 独立灯条通路只在上层提供跟踪 ROI 时运行，不影响网络装甲板检测。预处理与
  // 轮廓质量筛选参考 dx_vision 的传统装甲板检测：先按敌方颜色做 HSV 分割，
  // 再叠加亮度门限、轮廓填充率和几何门限，避免把白色反光和背景纹理当成灯条。
  // 宽高比定义为 width / length，与上面的 length / width 精修参数互为倒数。
  bool independent_light_enable{true};
  double independent_light_binary_threshold{120.0};
  double independent_light_threshold_tolerance{50.0};
  double independent_light_color_diff_threshold{20.0};
  double independent_light_min_contour_area_px{5.0};
  double independent_light_min_fill_ratio{0.5};
  float independent_light_min_length_px{6.0F};
  float independent_light_min_width_length_ratio{1.0F / 17.0F};
  float independent_light_max_width_length_ratio{1.0F / 2.4F};
  float independent_light_max_tilt_angle_deg{30.0F};

  // OpenCV HSV：H ∈ [0, 180]，S/V ∈ [0, 255]。默认值来自 dx_vision 的
  // preprocess_coefficient.yaml。红色跨越色相环零点，因此使用两段 H 区间。
  int independent_light_red_h_min_low{0};
  int independent_light_red_h_max_low{35};
  int independent_light_red_h_min_high{156};
  int independent_light_red_h_max_high{180};
  int independent_light_red_s_min{43};
  int independent_light_red_v_min{110};
  int independent_light_blue_h_min{80};
  int independent_light_blue_h_max{120};
  int independent_light_blue_s_min{30};
  int independent_light_blue_v_min{150};

  // dx_vision 默认关闭形态学；场地存在碎点时可开启。先闭运算连接灯条内部断点，
  // 再开运算清掉小噪点。核尺寸必须为正数。
  bool independent_light_use_morphology{false};
  int independent_light_morphology_width{3};
  int independent_light_morphology_height{3};
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

  // 在跟踪器给出的 ROI 内独立检测灯条。reference_armors 只用于像 Awakening
  // 一样按本帧网络灯条亮度自适应二值化阈值，不参与灯条归属判断。
  std::vector<Light> detectLights(
    const cv::Mat& image, const cv::Rect& roi,
    const std::vector<Armor>& reference_armors = {},
    ArmorColor target_color = ArmorColor::Unknown) const;

  const ArmorRefinerConfig& config() const noexcept { return config_; }

private:
  bool detectOne(Armor& armor, const cv::Mat& bgr_img, RefineRecord* record) const;

  ArmorRefinerConfig config_;
};

}  // namespace L2Perception
