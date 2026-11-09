#pragma once

namespace runtime {

// A tracked ROI needs the exposure pose before detection. Full-frame acquisition
// can run first while the serial RX thread fills pose history. resolve_pose must
// use the deadline established at frame read, not start a new timeout.
template <class ResolvePose, class Detect>
auto detectWithPose(bool needs_pose_for_roi, ResolvePose&& resolve_pose, Detect&& detect)
{
  if (needs_pose_for_roi) resolve_pose();
  auto result = detect();
  if (!needs_pose_for_roi) resolve_pose();
  return result;
}

}  // namespace runtime
