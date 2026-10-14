#pragma once

#include "rune/types.hpp"

namespace Rune {

class BuildingStage {
public:
  explicit BuildingStage(Geometry geometry) : geometry_(geometry) {}
  [[nodiscard]] std::optional<Pose> run(const Observation& observation,
                                        const Frame& frame) const;

private:
  Geometry geometry_;
};

}  // namespace Rune
