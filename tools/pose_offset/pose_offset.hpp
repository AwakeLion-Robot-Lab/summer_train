#pragma once

#include <Eigen/Core>

#include <optional>
#include <vector>

// 图像时间戳与云台姿态时间戳之间固定偏移的拟合，只依赖 Eigen，不碰相机和串口，
// 所以能用合成数据单测（tests/pose_offset_smoke.cpp）。
//
// 原理：静止目标在世界系里的视线方向本应恒定。姿态查询时刻比真值错开 Δτ，相机
// 朝向就偏 ω·Δτ，视线跟着云台角速度 ω 来回摆，于是视线方向的离散度约为
//
//   cost(τ) ≈ σ²(检测噪声) + (τ − τ*)²·Var(ω⊥)
//
// 扫一组 τ，最小处就是 τ*。Var 是方差而不是均方：云台匀速朝一个方向转时，偏移
// 只造成一个恒定的视线偏差，离散度不随 τ 变，τ* 不可观。所以标定时云台必须来回
// 甩，OffsetFit::rate_std 量的就是这一点。
namespace tools {

// 一帧样本：同一块静止装甲板在每个候选偏移下算出的世界系位置，下标与
// offsets_ms 一一对应。
using OffsetFrame = std::vector<Eigen::Vector3d>;

struct OffsetFit {
  // 每个候选偏移下视线方向的离散度，rad²，去掉最大的 10% 后取均值。
  std::vector<double> cost;
  // 网格最小值附近三点抛物线细化后的偏移，ms。
  double best_ms{0.0};
  // 只用前一半、后一半样本各拟合一次。两者差得多说明结果被噪声或目标移动带偏。
  double first_ms{0.0};
  double second_ms{0.0};
  // 视线角速度（垂直于视线的分量）的标准差，rad/s，即激励强度。
  double rate_std{0.0};
  // 最优偏移处剩下的视线离散度开方，deg，即检测与 PnP 的噪声底。
  double floor_deg{0.0};
  int frames{0};
  // 最小值落在扫描边界，真值可能在范围之外。
  bool edge{false};
};

// offsets_ms 必须等间距递增且至少 3 个；每帧的位置数必须与之相同，否则该帧
// 被跳过。可用样本少于 20 帧时返回空。
std::optional<OffsetFit> fitOffset(
  const std::vector<double>& offsets_ms, const std::vector<OffsetFrame>& frames);

}  // namespace tools
