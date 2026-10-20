#include "l3_estimation/buff/energy_fitter.hpp"
#include "l1_sensor/buff/transform.hpp"
#include "l4_planning/buff/attack_scheduler.hpp"
#include "l4_planning/buff/trajectory_solver.hpp"
#include "l5_control/buff/controller.hpp"

#include <Eigen/Core>

#include <chrono>
#include <cmath>
#include <iostream>

int main()
{
  L3Estimation::Buff::EnergyFitter fitter;
  for (int i = 0; i <= 20; ++i) {
    const double t = i * 0.1;
    fitter.push(t, 2.0 * t + 0.1);
  }
  const auto linear = fitter.fitLinear();
  if (!linear || std::abs(linear->speed - 2.0) > 1e-6) {
    std::cerr << "energy fitter smoke failed\n";
    return 1;
  }

  L4Planning::Buff::TrajectorySolver trajectory;
  const auto solution = trajectory.solve(Eigen::Vector3d{3.0, 0.0, 0.0}, 22.5);
  if (!solution || !(solution->fly_time > 0.0) || std::abs(solution->yaw) > 1e-12) {
    std::cerr << "trajectory solver smoke failed\n";
    return 1;
  }

  using Clock = std::chrono::steady_clock;
  L4Planning::Buff::AttackScheduler scheduler{0.4, 0.2};
  const auto start = Clock::now();
  scheduler.start(start);
  if (scheduler.tick(start) || !scheduler.tick(start + std::chrono::milliseconds{400})) {
    std::cerr << "attack scheduler smoke failed\n";
    return 1;
  }

  L5Control::Buff::Controller controller;
  if (controller.safeHold()) {
    std::cerr << "empty buff hold must not emit world yaw=0\n";
    return 1;
  }

  // PnP 输出已是 x前/y左/z上；纯光学轴外参必须与其内部换轴抵消，不能
  // 再把面前目标旋转 90°。
  L1Sensor::CameraCalibration calibration;
  Eigen::Isometry3d optical_to_barrel = Eigen::Isometry3d::Identity();
  optical_to_barrel.linear() << 0.0, 0.0, 1.0,
                                -1.0, 0.0, 0.0,
                                 0.0, -1.0, 0.0;
  calibration.T_barrel_camera = optical_to_barrel;
  const Eigen::Quaterniond q_world_barrel{
    Eigen::AngleAxisd{0.7, Eigen::Vector3d::UnitZ()}};
  const auto camera_pose = L1Sensor::Buff::composeCameraPose(
    q_world_barrel, calibration);
  if (!camera_pose ||
      !camera_pose->q_world_camera.toRotationMatrix().isApprox(
        q_world_barrel.toRotationMatrix(), 1e-12)) {
    std::cerr << "buff camera pose applied the optical-axis conversion twice\n";
    return 1;
  }
  L4Planning::Buff::Plan plan;
  plan.valid = true;
  plan.yaw = 1.2;
  plan.pitch = -0.1;
  const auto command = controller.update(plan);
  const auto hold = controller.safeHold();
  if (!command || !hold || hold->yaw != plan.yaw || hold->pitch != plan.pitch || hold->shoot) {
    std::cerr << "buff hold must preserve the last valid angle with fire disabled\n";
    return 1;
  }
  std::cout << "Buff math smoke test passed\n";
  return 0;
}
