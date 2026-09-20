#pragma once

#include "l2_perception/armor.hpp"

#include <array>
#include <cstddef>
#include <vector>

#include <opencv2/core.hpp>

namespace L2Perception
{

// 一块板的精修结果。精修从不删除网络检出，失败就原样保留网络角点。
enum class RefineVerdict
{
  Refined,     // 找到左右两根灯条，角点已被替换
  NetworkKept  // 传统检测没找到可信的两根灯条，保留网络角点
};

// 参数默认值与 sp_vision_25-main/configs/standard3.yaml 的传统识别参数一致。
struct ArmorRefinerConfig
{
  bool enable{true};

  // 板 ROI 内的灰度二值化阈值。场地偏暗调低，曝光偏亮调高。
  double binary_threshold{150.0};
  float min_lightbar_length_px{8.0F};
  // 灯条偏离竖直方向的最大角度，单位为度。
  float max_angle_error_deg{45.0F};
  // 灯条 长 / 宽 的区间。
  float min_lightbar_ratio{1.5F};
  float max_lightbar_ratio{20.0F};

  // 左右灯条端点到网络角点的距离和必须小于它，传统角点才会覆盖网络角点。
  // SP Detector::detect(Armor&, image) 里写死为 15 px。
  float max_endpoint_distance_px{15.0F};

  // 灯条端点改用亮度梯度 + PCA 主轴定位，而不是 minAreaRect 的二值边界。
  //
  // minAreaRect 的端点由阈值决定：阈值偏低斑点膨胀、偏高收缩，误差方向正好
  // 沿灯条轴向，而灯条长度正是端点观测里最敏感的深度线索。PCA 版只用阈值圈
  // ROI，端点由灰度突变找，对曝光不敏感。
  //
  // 在几何筛选之后、选灯条之前执行，所以 max_endpoint_distance_px 那道门限
  // 看到的也是修正后的端点。
  bool pca_corner_correction{true};

  // 二值化与梯度搜索的底图：false 是灰度（sp_vision 原版），true 是颜色差分
  // （蓝板 B−R、红板 R−B，取该板自己的 Armor::color，Unknown 时退回灰度）。
  //
  // 默认 true，回放实测的结论，八段全过，见 auto_aim.yaml 的 refiner 注释。
  // 机理：灯条过曝后核心 B=R=255，差分归零，所以差分图切到的是灯条彩色的
  // 边缘而不是饱和白核——白核边界是曝光的产物，彩色边缘才接近真实发光边界。
  //
  // 阈值语义与灰度完全不同，单独给一个键，不复用 binary_threshold。
  bool color_channel_diff{true};
  double color_diff_threshold{50.0};
};

// 一次 refine() 的统计。
struct RefineStats
{
  std::size_t refined{0};
  std::size_t network_kept{0};
  // network_kept 的三种成因，用来区分"精修提了个坏建议被门限拦下"和"传统
  // 检测在这张底图上压根没找到灯条"——两者的触发率一样低，含义完全相反。
  std::size_t no_lightbar{0};
  std::size_t size_skipped{0};
  std::size_t shift_rejected{0};
  // no_lightbar 再往下拆：二值图里一共找到几个轮廓、各道形状筛选各毙掉几个。
  // 区分"底图上根本没有连通域"和"有连通域但形状不像灯条"——差分图把灯条切成
  // 空心环时属于后者，调阈值能救；属于前者就只能换底图。
  std::size_t contour_total{0};
  std::size_t rej_angle{0};
  std::size_t rej_ratio{0};
  std::size_t rej_length{0};
  std::size_t bar_kept{0};
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
  // 网络角点的平均灯条长度过短，传统检测不值得信。
  bool size_skipped{false};
  // 找到了灯条，但端点离网络角点太远。
  bool shift_rejected{false};
  // ROI 越界，按 SP 的做法直接放弃。
  bool out_of_image{false};
  float aspect_ratio{0.0F};
  float lightbar_length{0.0F};
  float corner_shift{0.0F};
  // 二值图里的轮廓数和各道形状筛选的毙掉数，见 RefineStats 同名字段。
  std::size_t contour_total{0};
  std::size_t rej_angle{0};
  std::size_t rej_ratio{0};
  std::size_t rej_length{0};
  std::size_t bar_kept{0};
};

// 在网络检出的板 ROI 内跑一次 SP-Vision 的传统灯条检测，找到可信的左右两根
// 灯条时用它们的端点替换网络角点。
class ArmorRefiner
{
public:
  explicit ArmorRefiner(ArmorRefinerConfig config = {});

  // 与 SP-Vision Detector::detect(Armor&, const cv::Mat&) 相同的单板接口。
  // 成功时就地替换 armor.corners 并返回 true；失败时 armor 保持不变。
  bool detect(Armor& armor, const cv::Mat& bgr_img) const;

  // 批量接口，逐块调用 detect。records 非空时逐块记下判定明细。
  RefineStats refine(
    const cv::Mat& image, std::vector<Armor>& armors,
    std::vector<RefineRecord>* records = nullptr) const;

  const ArmorRefinerConfig& config() const noexcept { return config_; }

private:
  bool detectOne(Armor& armor, const cv::Mat& bgr_img, RefineRecord* record) const;

  ArmorRefinerConfig config_;
};

}  // namespace L2Perception
