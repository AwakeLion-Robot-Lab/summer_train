#pragma once
#include "rune/types.hpp"
#include "rune/f2-building/plane_window.hpp"
#include <array>

namespace Rune {
class BuildingStage {
public:
  explicit BuildingStage(Geometry geometry);
  [[nodiscard]] std::optional<Pose> run(const Observation& observation, const Frame& frame);
  [[nodiscard]] std::optional<Pose> run(const std::vector<Observation>& observations, const Frame& frame);
  void reset();
private:
  struct CachedModel {
    std::array<std::vector<cv::Point3d>,3> boundaries;
    std::vector<cv::Point3d> pca_box;
  };
  [[nodiscard]] std::optional<Pose> solve(const std::vector<Observation>& observations,
                                         const Frame& frame, std::size_t reference,
                                         bool allow_pca = true);
  Geometry geometry_;
  std::vector<cv::Point3d> anchors_;
  std::array<std::optional<CachedModel>,3> models_;
  PlaneWindow plane_window_;
  std::optional<Pose> previous_;
  std::optional<Frame> previous_frame_;
};
} // namespace Rune
