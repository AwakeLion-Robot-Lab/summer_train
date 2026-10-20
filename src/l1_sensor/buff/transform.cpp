#include "l1_sensor/buff/transform.hpp"

namespace L1Sensor::Buff {
namespace {

// camera_ros (x前/y左/z上) -> OpenCV optical (x右/y下/z前)。
const Eigen::Matrix3d kRosCameraToOpenCv =
  (Eigen::Matrix3d() <<
    0.0, -1.0, 0.0,
    0.0, 0.0, -1.0,
    1.0, 0.0, 0.0).finished();

}  // namespace

std::optional<CameraPose> composeCameraPose(
  const Eigen::Quaterniond& q_world_barrel,
  const CameraCalibration& calibration) noexcept
{
  if (!calibration.T_barrel_camera || !q_world_barrel.coeffs().allFinite()) {
    return std::nullopt;
  }
  const Eigen::Isometry3d& barrel_camera = *calibration.T_barrel_camera;
  if (!barrel_camera.matrix().allFinite()) return std::nullopt;
  CameraPose result;
  // 标定给的是 optical->barrel，而 PnP 输出已是 camera_ros。两者先在 L1
  // 边界组合成 camera_ros->barrel；纯轴向安装时乘积应严格为单位阵。
  const Eigen::Matrix3d barrel_camera_ros =
    barrel_camera.linear() * kRosCameraToOpenCv;
  result.q_world_camera = (q_world_barrel *
    Eigen::Quaterniond{barrel_camera_ros}).normalized();
  result.t_world_camera = q_world_barrel * barrel_camera.translation();
  if (!result.q_world_camera.coeffs().allFinite() ||
      !result.t_world_camera.allFinite()) return std::nullopt;
  return result;
}

}  // namespace L1Sensor::Buff
