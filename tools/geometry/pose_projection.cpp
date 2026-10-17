// Extracted from the rune builder; algorithm reference: SZURPVision/RP-26Rune.
// Provenance and MIT license: ../../src/rune/f2-building/README.md and RP-26Rune-LICENSE.
#include "pose_projection.hpp"

#include <opencv2/calib3d.hpp>
#include <cmath>

namespace tools::geometry {

bool finite(const cv::Point3d& p) {
  return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
}
cv::Matx33d rotation(const PoseParameters& p) {
  cv::Matx33d r;
  cv::Rodrigues(cv::Vec3d(p[0],p[1],p[2]), r);
  return r;
}
bool inFront(const std::vector<cv::Point3d>& model, const PoseParameters& p) {
  if (!p.allFinite()) return false;
  const auto r = rotation(p);
  for (const auto& v : model)
    if ((r*cv::Vec3d(v.x,v.y,v.z))[2]+p[5] <= 1e-6) return false;
  return true;
}
std::vector<cv::Point2d> project(const std::vector<cv::Point3d>& model,
                               const PoseParameters& p, const cv::Mat& k, const cv::Mat& d) {
  std::vector<cv::Point2d> result;
  cv::projectPoints(model, cv::Vec3d(p[0],p[1],p[2]),
                    cv::Vec3d(p[3],p[4],p[5]), k, d, result);
  return result;
}
bool rigid(const Eigen::Isometry3d& t) {
  return t.matrix().allFinite() &&
    (t.linear().transpose()*t.linear()-Eigen::Matrix3d::Identity()).norm()<1e-6 &&
    std::abs(t.linear().determinant()-1)<1e-6 &&
    (t.matrix().row(3)-Eigen::RowVector4d(0,0,0,1)).norm()<1e-9;
}

} // namespace tools::geometry
