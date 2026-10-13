#include "tools/pose_offset/pose_offset.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>

namespace tools {
namespace {

constexpr int kMinFrames = 20;
// 每个候选偏移下去掉离散最大的这一比例，挡住偶发的误检和 PnP 镜像解。各候选
// 去掉的比例相同，代价曲线才连续。
constexpr double kTrim = 0.1;

using Frames = std::vector<OffsetFrame>;

// [begin, end) 这段样本在第 k 个候选偏移下的视线离散度，rad²。小角度下单位向量
// 的弦长平方约等于夹角平方。
double spread(const Frames& frames, std::size_t begin, std::size_t end, std::size_t k)
{
  Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  for (std::size_t i = begin; i < end; ++i) {
    mean += frames[i][k].normalized();
  }
  mean.normalize();

  std::vector<double> squared;
  squared.reserve(end - begin);
  for (std::size_t i = begin; i < end; ++i) {
    squared.push_back((frames[i][k].normalized() - mean).squaredNorm());
  }
  std::sort(squared.begin(), squared.end());
  const auto kept = std::max<std::size_t>(
    1, static_cast<std::size_t>(std::floor(squared.size() * (1.0 - kTrim))));
  double sum = 0.0;
  for (std::size_t i = 0; i < kept; ++i) {
    sum += squared[i];
  }
  return sum / static_cast<double>(kept);
}

std::vector<double> curve(
  const Frames& frames, std::size_t begin, std::size_t end, std::size_t count)
{
  std::vector<double> cost(count);
  for (std::size_t k = 0; k < count; ++k) {
    cost[k] = spread(frames, begin, end, k);
  }
  return cost;
}

// 网格最小值，再用它和左右邻点拟抛物线细化到网格之间。
double argmin(const std::vector<double>& offsets_ms, const std::vector<double>& cost, bool& edge)
{
  const auto k = static_cast<std::size_t>(
    std::min_element(cost.begin(), cost.end()) - cost.begin());
  edge = k == 0 || k + 1 == cost.size();
  if (edge) {
    return offsets_ms[k];
  }
  const double left = cost[k - 1];
  const double mid = cost[k];
  const double right = cost[k + 1];
  const double curvature = left - 2.0 * mid + right;
  const double step = offsets_ms[k + 1] - offsets_ms[k];
  if (curvature <= 0.0) {
    return offsets_ms[k];
  }
  return offsets_ms[k] + 0.5 * (left - right) / curvature * step;
}

// 垂直于视线的角速度的标准差。取中间相邻两个候选做差分：它们只差一个步长，
// 视线方向之差除以步长就是此刻的视线角速度。
double rateStd(const Frames& frames, const std::vector<double>& offsets_ms)
{
  const std::size_t k = offsets_ms.size() / 2 - 1;
  const double step_s = (offsets_ms[k + 1] - offsets_ms[k]) * 1e-3;
  std::vector<Eigen::Vector3d> rates;
  rates.reserve(frames.size());
  Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  for (const auto& frame : frames) {
    rates.push_back((frame[k + 1].normalized() - frame[k].normalized()) / step_s);
    mean += rates.back();
  }
  mean /= static_cast<double>(rates.size());
  double sum = 0.0;
  for (const auto& rate : rates) {
    sum += (rate - mean).squaredNorm();
  }
  return std::sqrt(sum / static_cast<double>(rates.size()));
}

}  // namespace

std::optional<OffsetFit> fitOffset(
  const std::vector<double>& offsets_ms, const std::vector<OffsetFrame>& frames)
{
  const std::size_t count = offsets_ms.size();
  if (count < 3) {
    return std::nullopt;
  }

  Frames valid;
  valid.reserve(frames.size());
  for (const auto& frame : frames) {
    const bool usable = frame.size() == count &&
      std::all_of(frame.begin(), frame.end(), [](const Eigen::Vector3d& p) {
        return p.allFinite() && p.norm() > 1e-6;
      });
    if (usable) {
      valid.push_back(frame);
    }
  }
  if (valid.size() < static_cast<std::size_t>(kMinFrames)) {
    return std::nullopt;
  }

  OffsetFit fit;
  fit.frames = static_cast<int>(valid.size());
  fit.cost = curve(valid, 0, valid.size(), count);
  fit.best_ms = argmin(offsets_ms, fit.cost, fit.edge);

  const std::size_t half = valid.size() / 2;
  bool ignored = false;
  fit.first_ms = argmin(offsets_ms, curve(valid, 0, half, count), ignored);
  fit.second_ms = argmin(offsets_ms, curve(valid, half, valid.size(), count), ignored);

  fit.rate_std = rateStd(valid, offsets_ms);
  fit.floor_deg =
    std::sqrt(*std::min_element(fit.cost.begin(), fit.cost.end())) * 180.0 / std::numbers::pi;
  return fit;
}

}  // namespace tools
