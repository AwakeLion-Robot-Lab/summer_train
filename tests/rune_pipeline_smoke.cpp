#include "rune/processor.hpp"

#include <stdexcept>
#include <chrono>
#include <cmath>
#include <memory>

namespace {
void check(bool value) { if (!value) throw std::runtime_error("rune pipeline check failed"); }
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
  check(no_target.status == Status::NoDetection && !no_target.found &&
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

  PlanningConfig config;
  config.small_warmup_s=config.big_warmup_s=0;
  config.mpc.enable_mpc=false;
  PlanningStage planner(config);
  FireControlConfig fire;
  fire.shoot_enable=true; fire.aim_settle_s=fire.idle_duration_s=0;
  fire.require_rearm=true; fire.min_shot_interval_s=0;
  ControlStage controller(fire);
  const auto plan=planner.plan(motion,Mode::Small,input);
  auto first=controller.update(plan,input);
  check(first.found && !first.fire && first.status==Status::InvalidFeedback);
  input.actual_yaw_rad=plan.yaw; input.actual_pitch_rad=plan.pitch;
  const auto shot=controller.update(plan,input);
  check(shot.found && shot.fire && shot.status==Status::Ready);
  check(!controller.update(plan,input).fire);
  controller.rearm();
  check(controller.update(plan,input).fire);
  input.bullet_speed_m_s=0;
  const auto bad=planner.plan(motion,Mode::Small,input);
  check(!bad.valid && bad.status==Status::BallisticFailed);
  input.bullet_speed_m_s=28;
  input.command_at=now+std::chrono::seconds(1);
  check(!planner.plan(motion,Mode::Small,input).valid);
  input.command_at=now+std::chrono::milliseconds(20);
  const auto big=planner.plan(motion,Mode::Big,input);
  check(big.valid && big.center_hold && !big.fire_permitted);
  EstimatorStage big_estimator({},std::make_unique<FakeBigModel>());
  const auto fitted_big=big_estimator.update(pose,Mode::Big);
  const auto big_plan=planner.plan(fitted_big,Mode::Big,input);
  check(big_plan.valid && !big_plan.center_hold && big_plan.fire_permitted);

  EstimatorStage estimator;
  check(!estimator.update(pose, Mode::Small).ready);
  pose.captured_at += std::chrono::milliseconds(50);
  pose.phase_rad = 0.05;
  check(!estimator.update(pose, Mode::Small).ready);
  pose.captured_at += std::chrono::milliseconds(50);
  pose.phase_rad = 0.1;
  check(!estimator.update(pose, Mode::Small).ready);
  for (int i=3;i<=32;++i) {
    pose.captured_at += std::chrono::milliseconds(50);
    pose.phase_rad = .05*i;
    const auto fitted=estimator.update(pose, Mode::Small);
    if (i==32) check(fitted.ready && fitted.model==MotionModel::Linear);
  }
  check(!estimator.update(pose, Mode::Big).ready);
}
