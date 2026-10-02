#include "l3_estimation/types.hpp"

namespace L3Estimation {

std::string toString(TrackState state)
{
  switch (state) {
  case TrackState::Lost:
    return "lost";
  case TrackState::Detecting:
    return "detecting";
  case TrackState::Tracking:
    return "tracking";
  case TrackState::TempLost:
    return "temp_lost";
  }
  return "unknown";
}

}  // namespace L3Estimation
