#pragma once

#include "l3_estimation/target_estimator.hpp"
#include "l4_planning/types.hpp"

#include <Eigen/Core>

#include <vector>

namespace L4Planning {

enum class ArmorType {
  Small,
  Large
};

// 将一份 L3 车辆状态预测到绝对命中时刻。
struct PredictionRequest {
  L3Estimation::TargetState target; // L3 发布时刻的整车状态
  TimePoint target_time{};          // 需要预测到的绝对时刻
};

// 单块装甲板在预测时刻的运动状态。
struct ArmorPose {
  int robot_id{-1};                         // 所属车辆编号
  int armor_id{-1};                         // 车辆上的装甲板编号，范围 0~3
  ArmorType armor_type{ArmorType::Small};   // 大/小装甲类型

  // 世界坐标系下的位置、速度和装甲板法向角。
  Eigen::Vector3d position_world{Eigen::Vector3d::Zero()};
  Eigen::Vector3d velocity_world{Eigen::Vector3d::Zero()};
  double yaw_world{0.0};

  TimePoint timestamp{}; // 该预测状态对应的绝对时刻
  bool valid{false};     // 所有预测量是否有限且模型输入合法
};

// 整车预测状态以及由它展开得到的四块候选装甲板。
struct PredictionResult {
  L3Estimation::TargetState predicted_vehicle;
  std::vector<ArmorPose> armor_candidates;
  bool valid{false};
};

class Predictor {
public:
  // 使用匀速、匀角速度模型将车辆状态向前预测 dt 秒。
  L3Estimation::TargetState predict(const L3Estimation::TargetState& target, double dt) const;
  // 预测到绝对时刻，并生成四块装甲板的位置、速度和朝向。
  [[nodiscard]] PredictionResult predict(const PredictionRequest& request) const;
};

}  // namespace L4Planning
