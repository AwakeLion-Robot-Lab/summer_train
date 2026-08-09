#pragma once

#include "l1_sensor/camera/camera_calibration.hpp"
#include "l2_perception/armor.hpp"

#include <Eigen/Core>

#include <array>
#include <chrono>
#include <cstdint>
#include <limits>
#include <optional>

namespace L3Estimation {

// L3 中的时间戳统一使用单调时钟，避免系统时间校准造成负时间差。
using TimePoint = std::chrono::steady_clock::time_point;
// 对外目标状态固定为九维，元素顺序见 Target::vector()。
using TargetVector = Eigen::Matrix<double, 9, 1>;
using TargetCovariance = Eigen::Matrix<double, 9, 9>;
using ArmorName = L2Perception::ArmorClass;

// 物理装甲板板型只决定 PnP 几何尺寸，车辆类别由 Armor::name 单独表示。
enum class ArmorType : std::uint8_t {
  Small,  // 小装甲板
  Big     // 大装甲板
};

// 识别类别 → 实际板型。场上只有四板车，大装甲板仅英雄使用：平衡步兵已不存在，
// 基地虽然有 Bs/Bb 两个类别但装甲板实物都是小板，所以只有 Hero 走 Big 分支。
//
// 未知类别返回 nullopt，不猜板型：猜错会同时污染 PnP 几何和火控的角度容差。
// L3 的 PnpSolver 和 L5 的 FireDecider 共用这一份映射，不各写一份。
[[nodiscard]] constexpr std::optional<ArmorType> armorTypeOf(ArmorName name) noexcept
{
  switch (name) {
    case ArmorName::Hero:
      return ArmorType::Big;

    case ArmorName::Guard:
    case ArmorName::Engineer:
    case ArmorName::Infantry3:
    case ArmorName::Infantry4:
    case ArmorName::Infantry5:
    case ArmorName::Outpost:
    case ArmorName::BaseSmall:
    case ArmorName::BaseLarge:
      return ArmorType::Small;

    case ArmorName::Unknown:
      break;
  }
  return std::nullopt;
}

// Tracker 的四态生命周期。
enum class TrackState : std::uint8_t {
  Lost,       // 当前没有可用目标
  Detecting,  // 已发现目标，等待连续帧确认
  Tracking,   // 稳定跟踪
  TempLost    // 短时丢失，继续输出预测状态
};

// 不把多个质量条件压成一个 bool，便于分别记录失败原因。
struct ArmorQuality {
  bool pnp_ok{false};           // PnP 求解器返回成功
  bool reprojection_ok{false};  // 像素重投影误差未超过门限
  bool geometry_ok{false};      // 所有物理角点均位于相机前方
  bool finite{false};           // 输出位姿和误差均为有限值

  // 只有全部质量检查通过的观测才能进入目标跟踪器。
  [[nodiscard]] bool valid() const noexcept
  {
    return pnp_ok && reprojection_ok && geometry_ok && finite;
  }
};

// 相机内参及静态机械外参由 L1 持有，L3 只补充时间同步状态。
struct AimCalibration {
  // 内参、畸变参数以及 camera -> barrel 的静态外参。
  L1Sensor::CameraCalibration camera;
  // 图像曝光时刻与枪管姿态已经完成时间对齐。
  bool time_sync_ok{false};

  // 这里只检查 PnP 所需矩阵是否存在，矩阵数值由 PnpSolver 进一步验证。
  [[nodiscard]] bool intrinsicsOk() const noexcept
  {
    return !camera.camera_matrix.empty() &&
           !camera.distortion_coefficients.empty();
  }

  [[nodiscard]] bool trackingReady() const noexcept
  {
    return intrinsicsOk() && camera.barrelExtrinsicsReady() && time_sync_ok;
  }

  [[nodiscard]] bool fireReady() const noexcept
  {
    return trackingReady();
  }
};

// L3 对 L2 输出的 Armor 执行单板 PnP 和坐标变换后得到的观测。
struct Armor {
  // 分类信息。name 是车辆类别，type 是实际采用的物理板型。
  ArmorName name{ArmorName::Unknown};
  ArmorType type{ArmorType::Small};
  int class_id{-1};

  // 图像角点顺序固定为左上、右上、右下、左下，单位为 pixel。
  std::array<cv::Point2f, 4> points{};
  // L2 检测得到的四角点几何中心，单位为 pixel。
  cv::Point2f center{};

  // 平移量单位均为 meter。
  Eigen::Vector3d xyz_in_camera{Eigen::Vector3d::Zero()};
  Eigen::Vector3d xyz_in_barrel{Eigen::Vector3d::Zero()};
  Eigen::Vector3d xyz_in_world{Eigen::Vector3d::Zero()};
  // 固定顺序为 [yaw, pitch, roll]，采用 Rz(yaw)Ry(pitch)Rx(roll)。
  Eigen::Vector3d ypr_in_camera{Eigen::Vector3d::Zero()};
  Eigen::Vector3d ypr_in_barrel{Eigen::Vector3d::Zero()};
  Eigen::Vector3d ypr_in_world{Eigen::Vector3d::Zero()};
  // [方位角, 俯仰角, 距离]，角度单位为 radian，距离单位为 meter。
  Eigen::Vector3d ypd_in_world{Eigen::Vector3d::Zero()};
  Eigen::Vector3d ypd_in_barrel{Eigen::Vector3d::Zero()};

  // 四个角点的二维像素 RMSE，取自 IPPE 在相机系的原始解，与 yaw 优化无关。
  double reprojection_error{std::numeric_limits<double>::infinity()};
  // 检测置信度和四边形像素面积从 L2 原样传入。
  double confidence{0.0};
  double yaw_raw{0.0};
  double area{0.0};
  // yaw 优化收敛点的标准差，单位 radian，由高斯牛顿的曲率 J^T J 给出：
  // sigma_yaw^2 = corner_noise_px^2 / (J^T J)。正对装甲板时 yaw 几乎不可
  // 观测（转动几乎不改变投影），该值会显著变大，这正是固定方差表达不出来的
  // 信息。优化未收敛时保持无穷。
  //
  // L3 的 EKF **不**消费这个字段，update_ypda 里仍用手调的 armor_yaw_variance
  // （3 米处 sigma 约 17.8 度）。接过去试过了，是负面结果，别再照着直觉重试：
  //
  //   corner_noise_px=1 时接进 R，NIS 从 0.127 变成 4.799（四维观测的理论值就是
  //   4.0，说明这个噪声模型在统计上是自洽的），但 100 ms 开环预测误差从
  //   0.0553 涨到 0.0627 m。把 sigma 放大 6 倍才追平手调值（0.0555）。
  //   3m_low 上同样：NIS 3.623 而预测误差 0.0431 -> 0.0486 m。
  //
  //   关键证据是收紧板 yaw 会拖累**方位角**创新（3m_low 上 0.0246 -> 0.0769 度）。
  //   一个分量的噪声估计偏小不会伤到另一个独立分量，所以问题不是"估小了"，而是
  //   板 yaw 观测与位置观测在整车刚体模型下互相矛盾：曲率只描述"单块板的像素能把
  //   yaw 定到多准"，看不见角点系统偏差、reproject_armor 里写死的 15 度安装倾角、
  //   以及残留的 PnP 盆地误差。手调的宽 R 实际上是在压制这个矛盾，所以预测更好。
  //
  // 要让它有用，得先消掉矛盾源：打开灯条精修换掉网络角点，或把安装倾角变成标定量。
  double yaw_sigma{std::numeric_limits<double>::infinity()};

  ArmorQuality quality;
  // 对应原始图像的曝光时刻。
  TimePoint timestamp{};
};

// 跨层输出使用的九维整车状态：
// [xc, vx, yc, vy, z, vz, yaw, v_yaw, radius]。
struct Target {
  ArmorName name{ArmorName::Unknown};
  int target_id{-1};

  // 本帧**最后一个**被处理的观测关联到的物理装甲板编号。仅供调试，
  // 对应 sp_vision Target 里那个标了 "debug only" 的 last_id。
  //
  // 不要拿它选板、也不要拿它画曲线。Tracker::updateTarget 对每个同类观测各调
  // 一次 update()，这个字段留的是最后一次的结果，而观测按到图像中心的距离排序
  // ——两块板都可见时它报的是**离画面中心更远**的那块，可见板数在 1 和 2 之间
  // 变化时它就跟着变。实测 records/3m_run_mid：159 次变化里 134 次只是因为第二
  // 块板出现或消失，其中 30 次是"切过去又切回"。
  //
  // 该瞄哪块板由 L4 回答：Planner::selectArmor 在命中时刻的整车几何上选，带
  // front_window 和 switch_hysteresis，结果在 Plan::armor_id。同一段录像上它
  // 只切 129 次，没有抖动。
  int armor_id{-1};

  // 旋转中心在世界坐标系中的位置和速度，单位分别为 meter、meter/second。
  Eigen::Vector3d position{Eigen::Vector3d::Zero()};
  Eigen::Vector3d velocity{Eigen::Vector3d::Zero()};
  // 整车中心 yaw、角速度以及当前主半径，单位为 radian、radian/second、meter。
  double yaw{0.0};
  double v_yaw{0.0};
  double radius{0.0};

  // 整车几何的其余部分。L4 需要靠这三个字段自行展开全部物理装甲板并外推到
  // 命中时刻，否则只能拿到当前帧的装甲板位置，无法做延迟补偿和选板。
  // second_radius 是四板车奇数板使用的另一组半径（已含 r2-r1），height_diff
  // 是对应的高度差 z2-z1；三板车两者与主半径、0 相同。
  int armor_num{4};
  double second_radius{0.0};
  double height_diff{0.0};

  // 是否已经关联到过 0 号以外的装甲板。为 false 时整车 yaw、第二组半径和
  // 高度差几乎不可观测——只见过一块板的话，其余板的位置完全由初值猜出来。
  // L4 必须据此只瞄当前观测到的那块板，否则等于拿伪造的几何去开火，违反
  // "缺失标定保持缺失"的约定。一旦观测到过就保持为 true。
  bool multi_armor_observed{false};

  // P 与 vector() 使用完全相同的九维元素顺序。
  TargetCovariance P{TargetCovariance::Identity()};
  TrackState track_state{TrackState::Lost};
  TimePoint timestamp{};

  // nis 是最近一次滤波更新的创新统计量；updated 表示本帧使用了观测更新。
  double nis{0.0};
  bool updated{false};

  [[nodiscard]] TargetVector vector() const noexcept
  {
    TargetVector x;
    x << position.x(), velocity.x(), position.y(), velocity.y(), position.z(),
      velocity.z(), yaw, v_yaw, radius;
    return x;
  }
};

struct ArmorConfig {
  // 装甲板几何尺寸，单位为 meter。
  double small_width{0.135};
  double big_width{0.230};
  double height{0.056};
  // 单位分别为 pixel RMSE 和 pixel²。
  double max_reprojection_error{3.0};
  double min_area{20.0};
  // 单个角点坐标的像素噪声标准差，用于把 yaw 搜索的曲率换算成 Armor::yaw_sigma。
  // 1.0 是标称值，不是实测值：真正的角点噪声取决于网络回归精度和是否做过灯条
  // 精修，改动这里等于整体缩放 yaw_sigma，接入 EKF 之前必须先标定。
  double corner_noise_px{1.0};
};

// sp_vision 行为复刻开关。
//
// 用途是差分定位，不是"另一套调参"：把 L3 逐项切回 sp_vision 的实现——**包括
// 它已知的 bug**——先确认 newvision 能复现 sp 的行为，再一项项关掉，看哪一项
// 才是差异的来源。所以每个字段都必须严格等于 sp 的实现，不许"顺手改好一点"，
// 否则复现失败时分不清是没抄对还是结论不成立。
//
// enable 为 false 时全部字段无效，走 newvision 自己的实现。
struct SpCompatConfig {
  bool enable{false};

  // 观测噪声用 sp 的数值：方位角/俯仰角 4e-3（sigma 3.62 度，实测噪声的 13
  // 倍），距离 log1p(|delta_angle|) + 1.0（正视时 sigma 就有 1 米）。
  // newvision 现在是 1e-4 和 2.5e-3 + log1p(|delta_angle|)。
  bool loose_observation_noise{true};

  // x_add 里不把 r1 和 r2 投影回 [0.05, 0.5]。sp 不做这个投影，任由半径被
  // 单次观测拽出物理范围，再靠 diverged() 把整个目标丢掉。
  bool unclamped_radius{true};

  // diverged() 用 sp 的瞬时判据：r 或 r2 一旦越界立刻判发散。newvision 是
  // "连续贴边 10 次"，因为投影之后偶发越界已经不是发散信号。
  bool instant_divergence{true};

  // NIS 用 sp 的算法：后验残差 + 后验 P 组成的 S，门限固定 0.711。
  // 这个量恒偏小、不服从卡方分布，而且 0.711 是自由度 4 的**下** 5% 分位
  // （上分位是 9.488），所以 sp 的 "Bad Converge" 复位判据实际在拿一个错误
  // 的统计量比一个错误的门限。Tracker 的复位行为直接受它影响，必须一起抄。
  bool posterior_nis{true};
};

struct TrackerConfig {
  // 从 Detecting 转入 Tracking 所需的连续有效观测帧数。
  int min_detect_count{5};
  // 非 Lost 状态允许的最大相邻帧间隔。
  std::chrono::milliseconds max_frame_interval{100};
  // 临时丢失按连续帧数计数；前哨站允许更长的无观测预测窗口。
  int max_temp_lost_count{15};
  int outpost_max_temp_lost_count{75};

  // 观测是否必须通过 ArmorQuality 的全部门限才能进入滤波。
  //
  // 只有离线回放调试才允许关掉：关掉后门限退化成"single_pnp 成功产出了位姿"，
  // 等于把未经重投影和角点可见性校验的 PnP 结果直接喂给 EKF，发散和跳变都属于
  // 预期内的现象。实机必须保持 true，否则一次坏解就能把整车状态带跑。
  bool require_quality{true};

  // sp_vision 行为复刻，见 SpCompatConfig。默认关闭，实机不要打开。
  SpCompatConfig sp_compat{};
};

// 跨层接口使用的语义别名。
using ArmorObservation = Armor;
using TargetState = Target;
using TrackStatus = TrackState;

}  // namespace L3Estimation
