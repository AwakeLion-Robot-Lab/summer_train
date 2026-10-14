#include "rune/processor.hpp"

#include <cassert>
#include <chrono>
#include <cmath>
#include <memory>

namespace {
class FakeBigModel final : public Rune::IBigMotionModel {
public:
  Rune::Motion update(const Rune::Pose& pose) override {
    Rune::Motion result;
    result.pose = pose;
    result.ready = true;
    result.phase_delta_rad = [](double seconds) -> std::optional<double> {
      return seconds;
    };
    return result;
  }
  void reset() override {}
};
}  // namespace

int main() {
  using namespace Rune;
  const auto now = std::chrono::steady_clock::now();

  Frame empty;
  empty.mode = Mode::Small;
  const auto no_target = Processor(Geometry{0.7, 0.07, 0.06})
    .process(empty, ControlInput{.command_at = now});
  assert(no_target.status == Status::NoDetection && !no_target.found &&
         !no_target.fire);

  Pose pose;
  pose.center_barrel = {8, 0, 0};
  pose.target_barrel = {8, 0, 0.7};
  pose.normal_barrel = {1, 0, 0};
  pose.captured_at = now;
  Motion motion{pose, 1.0, true};
  ControlInput input;
  input.command_at = now + std::chrono::milliseconds(20);
  input.bullet_speed_m_s = 28;
  input.fire_delay_s = 0.02;

  PlanningStage planner({.shoot_enable = true});
  const auto first = planner.plan(motion, Mode::Small, input);
  assert(first.found && !first.fire && first.status == Status::AimNotReady);
  assert(std::isfinite(first.yaw_rad) && std::isfinite(first.pitch_rad));

  input.actual_yaw_rad = first.yaw_rad;
  input.actual_pitch_rad = first.pitch_rad;
  const auto shot = planner.plan(motion, Mode::Small, input);
  assert(shot.found && shot.fire && shot.status == Status::Ready);
  const auto repeated = planner.plan(motion, Mode::Small, input);
  assert(repeated.found && !repeated.fire &&
         repeated.status == Status::AlreadyFired);
  planner.resetFireLatch();
  assert(planner.plan(motion, Mode::Small, input).fire);

  input.bullet_speed_m_s = 0;
  const auto bad_speed = planner.plan(motion, Mode::Small, input);
  assert(!bad_speed.found && !bad_speed.fire &&
         bad_speed.status == Status::BallisticFailed);
  input.bullet_speed_m_s = 28;
  input.command_at = now + std::chrono::seconds(1);
  const auto stale = planner.plan(motion, Mode::Small, input);
  assert(!stale.found && !stale.fire);
  input.command_at = now + std::chrono::milliseconds(20);
  const auto big = planner.plan(motion, Mode::Big, input);
  assert(!big.found && !big.fire);

  EstimatorStage big_estimator({}, std::make_unique<FakeBigModel>());
  const auto fitted_big = big_estimator.update(pose, Mode::Big);
  const auto big_aim = planner.plan(fitted_big, Mode::Big, input);
  assert(big_aim.found && !big_aim.fire);

  EstimatorStage estimator;
  assert(!estimator.update(pose, Mode::Small).ready);
  pose.captured_at += std::chrono::milliseconds(50);
  pose.phase_rad = 0.05;
  assert(!estimator.update(pose, Mode::Small).ready);
  pose.captured_at += std::chrono::milliseconds(50);
  pose.phase_rad = 0.1;
  assert(estimator.update(pose, Mode::Small).ready);
  assert(!estimator.update(pose, Mode::Big).ready);
}
