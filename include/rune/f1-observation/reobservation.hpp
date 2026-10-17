#pragma once

#include "rune/types.hpp"

namespace Rune {

class ObservationStage {
public:
  [[nodiscard]] std::optional<Observation> run(const Region& region,
                                                Color color,
                                                Mode mode = Mode::Small) const;
};

}  // namespace Rune
