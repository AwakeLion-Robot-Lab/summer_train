#include "l3_planning/tiny_mpc.hpp"
#include "l3_planning/armor_candidate_generator.hpp"
#include "l3_planning/in_plate_aim.hpp"
#include "l3_planning/planner.hpp"
#include "l3_planning/planner_config.hpp"
#include "l4_control/fire_decision.hpp"
#include "runtime/l4_target_adapter.hpp"

#include <Eigen/Geometry>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <numbers>
#include <string_view>

using namespace L4Planning;
using namespace std::chrono_literals;
namespace {
int failures = 0;
void expect(bool value, std::string_view message)
{
  if (!value) { ++failures; std::cerr << "FAIL: " << message << '\n'; }
}

void projectionChecks()
{
  TinyMpcPlateRegion region;
  region.jacobian << 0.3, -0.12, 0.1, 0.25;
  region.limits = {0.03, 0.02};
  const Eigen::Vector2d weight{3.0, 1.0};
  const Eigen::Vector2d inside{0.01, -0.015};
  expect((projectTinyMpcPlate(region, region.jacobian * inside, weight) - inside).norm() < 1e-12,
    "tilted plate interior projection preserves coupled coordinates");
  for (const Eigen::Vector2d query : {Eigen::Vector2d{0.02, -0.02}, Eigen::Vector2d{-0.02, 0.01}}) {
    const auto uv = projectTinyMpcPlate(region, query, weight);
    expect((uv.array().abs() <= region.limits.array()).all(), "projection stays inside true plate rectangle");
    const Eigen::Vector2d error = region.jacobian * uv - query;
    const double cost = error.dot(weight.asDiagonal() * error);
    double grid_best = std::numeric_limits<double>::infinity();
    for (int u = -50; u <= 50; ++u) for (int v = -50; v <= 50; ++v) {
      const Eigen::Vector2d point{region.limits.x() * u / 50.0, region.limits.y() * v / 50.0};
      const Eigen::Vector2d grid_error = region.jacobian * point - query;
      grid_best = std::min(grid_best, grid_error.dot(weight.asDiagonal() * grid_error));
    }
    expect(cost <= grid_best + 1e-12, "weighted projection agrees with independent exhaustive grid");
  }
  region.limits.setZero();
  expect(projectTinyMpcPlate(region, {1.0, -1.0}, weight).norm() == 0.0, "collapsed region uses center");
  region.jacobian.setZero(); region.limits.setConstant(0.03);
  expect(projectTinyMpcPlate(region, {1.0, -1.0}, weight).allFinite(), "singular geometry stays finite");
}

void optimizerChecks()
{
  TinyMpcReference reference;
  reference.valid = true;
  reference.optimize_in_plate = true;
  for (std::size_t k = 0; k < kTinyMpcHorizon; ++k) {
    const double t = k * kTinyMpcStepSeconds;
    reference.yaw.angle[k] = 0.01 * (1.0 - std::cos(6.0 * t));
    reference.yaw.angular_velocity[k] = 0.06 * std::sin(6.0 * t);
    reference.regions[k].center = {reference.yaw.angle[k], 0.0};
    reference.regions[k].jacobian = Eigen::Matrix2d::Identity() * 0.25;
    reference.regions[k].limits = {0.035, 0.03};
  }
  PlannerConfig config;
  const auto solution = solveTinyMpc(reference, config);
  std::cout << "region solve: iterations=" << solution.yaw.info.iteration_count
    << " primal=" << solution.yaw.info.primal_residual
    << " dual=" << solution.yaw.info.dual_residual << '\n';
  expect(solution.valid, "joint optimizer converges");
  auto point_reference = reference;
  point_reference.optimize_in_plate = false;
  const auto point = solveTinyMpc(point_reference, config);
  expect(point.valid, "point baseline converges");
  double point_effort = 0.0, region_effort = 0.0, max_offset = 0.0;
  for (std::size_t k = 0; k < kTinyMpcHorizon; ++k) {
    const auto uv = solution.in_plate_offsets[k];
    expect((uv.array().abs() <= reference.regions[k].limits.array() + 1e-12).all(), "optimized offset bounded");
    expect(std::abs(solution.yaw.acceleration[k]) <= config.max_yaw_acceleration + 1e-12,
      "projected control respects physical acceleration bound");
    point_effort += point.yaw.acceleration[k] * point.yaw.acceleration[k];
    region_effort += solution.yaw.acceleration[k] * solution.yaw.acceleration[k];
    max_offset = std::max(max_offset, uv.norm());
    if (k + 1 < kTinyMpcHorizon) {
      expect(std::abs(solution.yaw.angle[k + 1] - solution.yaw.angle[k]
        - kTinyMpcStepSeconds * solution.yaw.angular_velocity[k]) < 1e-12, "angle dynamics preserved");
      expect(std::abs(solution.yaw.angular_velocity[k + 1] - solution.yaw.angular_velocity[k]
        - kTinyMpcStepSeconds * solution.yaw.acceleration[k]) < 1e-12, "velocity dynamics preserved");
    }
  }
  expect(max_offset > 0.005, "optimizer actively chooses a point away from center");
  std::cout << "effort point=" << point_effort << " region=" << region_effort << " offset=" << max_offset << '\n';
  expect(region_effort < point_effort * 0.8, "plate freedom reduces acceleration cost on curved trajectory");
  auto invalid = reference;
  invalid.regions[40].jacobian(0, 0) = std::numeric_limits<double>::quiet_NaN();
  expect(!solveTinyMpc(invalid, config).valid, "nonfinite region rejected");
  config.mpc_region_max_iterations = 1;
  expect(!solveTinyMpc(reference, config).valid, "unfinished optimization is not reported as valid");
}

AimPlan directPlan(const L3Estimation::TargetState& target, const L1Sensor::RobotState& robot,
  const PlannerConfig& config, InPlateAimState state)
{
  PlannerContext context;
  context.config = config;
  const auto fire_time = robot.timestamp + 15ms;
  const auto candidates = generateArmorCandidates(target, robot, context, fire_time, state, 0);
  AimPlan direct;
  if (!candidates || candidates->empty() || !candidates->front().valid) return direct;
  const auto& c = candidates->front();
  direct.valid = direct.tracked = direct.tracked_ready = true;
  direct.fire_permitted = direct.within_firing_window = c.within_firing_window;
  direct.tracked_phase = ArmorTrackingPhase::Tracking;
  direct.target_id = target.robot_id; direct.armor_id = 0;
  direct.generated_at = robot.timestamp; direct.fire_time = fire_time;
  direct.impact_time = c.impact_time; direct.fly_time = c.ballistic.fly_time;
  direct.yaw = c.ballistic.yaw; direct.pitch = c.ballistic.pitch;
  direct.aim_query_state = state; direct.in_plate_aim = c.aim;
  direct.aim_point_world = direct.aim_point_barrel = c.aim.position_world;
  direct.armor_center_world = direct.armor_center_barrel = c.armor.position_world;
  direct.armor_rotation_world = c.armor.rotation_world;
  direct.armor_geometry_valid = true; direct.bullet_speed = robot.bullet_speed;
  direct.gravity = config.gravity;
  direct.drag_coefficient = config.enable_air_resistance ? config.linear_drag_coefficient : 0.0;
  direct.hit_half_width = config.in_plate_aim.small_width * config.in_plate_aim.hit_margin_ratio / 2.0;
  direct.hit_half_height = config.in_plate_aim.height * config.in_plate_aim.hit_margin_ratio / 2.0;
  return direct;
}

void pipelineChecks()
{
  for (const bool outpost : {false, true}) for (const double rate : {2.51, -2.51}) {
    L3Estimation::TargetState target;
    target.robot_id = outpost ? static_cast<int>(L3Estimation::ArmorName::Outpost) : 3;
    target.armor_count = outpost ? 3 : 4;
    target.center = {4.0, 0.0, 0.2}; target.radius = 0.2765;
    target.yaw_rate = rate; target.height_offset = 0.04;
    target.yaw = -rate * 0.19; // 当前命中时刻处于正对段，另测临近换板的过渡。
    target.three_armor_height_offsets = {0.0, 0.01, 0.04};
    target.timestamp = TimePoint{2s}; target.covariance.setZero();
    L1Sensor::RobotState robot;
    robot.timestamp = target.timestamp + 10ms; robot.bullet_speed = 25.0;
    InPlateAimState state;
    state.robot_id = target.robot_id; state.armor_id = 0;
    state.timestamp = target.timestamp - 10ms; state.stable_frames = 10;
    PlannerConfig config;
    config.in_plate_aim.enable = true; config.in_plate_aim.mode = InPlateAimMode::Mpc;
    config.enable_air_resistance = true; config.linear_drag_coefficient = 0.5;
    const auto direct = directPlan(target, robot, config, state);
    const auto reference = buildTinyMpcReference(direct, target, robot, Eigen::Isometry3d::Identity(), config);
    const auto solve = solveTinyMpc(reference, config);
    std::cout << "pipeline solve: reference=" << reference.valid << " valid=" << solve.valid
      << " iterations=" << solve.yaw.info.iteration_count << " primal=" << solve.yaw.info.primal_residual
      << " dual=" << solve.yaw.info.dual_residual << '\n';
    expect(direct.valid && direct.in_plate_aim.offset.norm() == 0.0, "MPC direct fallback uses center");
    const auto before = std::chrono::steady_clock::now();
    const auto plan = applyTinyMpc(direct, target, robot, Eigen::Isometry3d::Identity(), config);
    std::cout << "pipeline outpost=" << outpost << " rate=" << rate << " mpc=" << plan.using_MPC
      << " hit=" << plan.mpc_aim_feasible << " offset=" << plan.in_plate_aim.offset.transpose()
      << " ms=" << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - before).count() << '\n';
    BallisticRequest req;
    req.target_position_barrel = plan.aim_point_barrel; req.bullet_speed = robot.bullet_speed;
    req.enable_air_resistance = true; req.linear_drag_coefficient = 0.5;
    const auto exact = BallisticSolver{}.solve(req);
    expect(plan.valid && plan.using_MPC, "MPC mode optimizes both rotation directions and plate counts");
    expect(plan.mpc_aim_feasible, "current command intersects safe plate in representative tracking cases");
    expect(std::abs(plan.yaw - exact.yaw) < 1e-5 && std::abs(plan.pitch - exact.pitch) < 1e-5,
      "nonlinear published aim and executed angles share the same ballistic solution");
    expect((plan.in_plate_aim.offset.array().abs() <= plan.in_plate_aim.limits.array() + 1e-12).all(),
      "published aim stays inside uncertainty adjusted plate");
    expect(std::abs(std::chrono::duration<double>(plan.impact_time - plan.fire_time).count()
      - plan.fly_time) < 1e-6, "published impact and flight time agree");
    if (plan.using_MPC) {
      expect(plan.samples.front().yaw == plan.yaw && plan.samples.front().pitch == plan.pitch,
        "command and published angles agree");
      for (const auto& sample : plan.samples) {
        expect(std::abs(sample.yaw_acceleration) <= config.max_yaw_acceleration + 1e-9
          && std::abs(sample.pitch_acceleration) <= config.max_pitch_acceleration + 1e-9, "all commands bounded");
      }
    }
    auto changed = config;
    changed.in_plate_aim.lookahead = 0.8; changed.in_plate_aim.response_time = 0.001;
    changed.in_plate_aim.max_speed = 0.001; changed.in_plate_aim.max_acceleration = 0.01;
    const auto other = applyTinyMpc(direct, target, robot, Eigen::Isometry3d::Identity(), changed);
    expect(other.using_MPC == plan.using_MPC && std::abs(other.yaw - plan.yaw) < 1e-12
      && std::abs(other.pitch - plan.pitch) < 1e-12, "MPC has no dependency on heuristic motion knobs");
    auto failed_config = config; failed_config.mpc_region_max_iterations = 1;
    const auto failed = applyTinyMpc(direct, target, robot, Eigen::Isometry3d::Identity(), failed_config);
    expect(!failed.using_MPC && failed.samples.empty()
      && (failed.aim_point_world - direct.aim_point_world).norm() == 0.0, "nonconvergence falls back to consistent direct plan");
    auto transition_target = target;
    transition_target.yaw = 0.0;
    const auto transition_direct = directPlan(transition_target, robot, config, state);
    const auto transition = applyTinyMpc(transition_direct, transition_target, robot,
      Eigen::Isometry3d::Identity(), config);
    expect(transition.using_MPC && !transition.mpc_aim_feasible && !transition.fire_permitted,
      "crossing a plate gap still produces motion but cannot permit firing");
  }
}

void candidateSeedChecks()
{
  L3Estimation::TargetState target;
  target.robot_id = 3; target.center = {4.0, 0.0, 0.2}; target.radius = 0.26;
  target.yaw_rate = 2.51; target.timestamp = TimePoint{2s}; target.covariance.setZero();
  L1Sensor::RobotState robot;
  robot.timestamp = target.timestamp; robot.bullet_speed = 25.0;
  PlannerContext context;
  context.config.enable_air_resistance = true; context.config.linear_drag_coefficient = 0.5;
  context.config.in_plate_aim.enable = true; context.config.in_plate_aim.mode = InPlateAimMode::Mpc;
  InPlateAimState state;
  state.robot_id = 3; state.armor_id = 0; state.stable_frames = 10; state.timestamp = target.timestamp;
  const auto cold = generateArmorCandidates(target, robot, context, target.timestamp, state, 0);
  expect(cold && cold->front().valid, "cold candidate for seed comparison converges");
  if (!cold || !cold->front().valid) return;
  auto seed = cold->front();
  seed.ballistic.fly_time += 0.003;
  seed.ballistic.pitch += 0.01;
  const auto warm = generateArmorCandidates(target, robot, context, target.timestamp, state, 0, std::nullopt, &seed);
  expect(warm && warm->front().valid
    && std::abs(warm->front().ballistic.fly_time - cold->front().ballistic.fly_time) < 0.0002
    && (warm->front().aim.position_world - cold->front().aim.position_world).norm() < 0.005,
    "warm start retains original impact convergence limits");
  seed.armor.robot_id = 4;
  const auto other = generateArmorCandidates(target, robot, context, target.timestamp, state, 0, std::nullopt, &seed);
  expect(other && other->front().valid
    && (other->front().aim.position_world - cold->front().aim.position_world).norm() == 0.0,
    "another target's seed is ignored");
  seed.armor.robot_id = 3; seed.armor.armor_id = 1;
  const auto another_plate = generateArmorCandidates(target, robot, context, target.timestamp, state, 0, std::nullopt, &seed);
  expect(another_plate && another_plate->front().valid
    && (another_plate->front().aim.position_world - cold->front().aim.position_world).norm() == 0.0,
    "plate handoff does not reuse another physical plate's seed");
  target.velocity = {10.0, 0.0, 0.0};
  const auto moving_cold = generateArmorCandidates(target, robot, context, target.timestamp, state, 0);
  seed.armor.armor_id = 0; seed.ballistic.fly_time = 10.0;
  const auto retried = generateArmorCandidates(target, robot, context, target.timestamp, state, 0, std::nullopt, &seed);
  expect(moving_cold && moving_cold->front().valid && retried && retried->front().valid
    && (retried->front().aim.position_world - moving_cold->front().aim.position_world).norm() == 0.0,
    "unreachable seeded prediction retries original cold-start path");
}

void filterAndFireChecks()
{
  for (const double drag : {0.0, 0.5}) {
    L5Control::FireInput input;
    input.target = L3Estimation::EskfTarget(L3Estimation::ArmorName::Infantry3, 4.0, 0.0, 0.26);
    input.target->predict(TimePoint{2s});
    const auto target = *runtime::toL4TargetState(input.target);
    L1Sensor::RobotState robot;
    robot.timestamp = target.timestamp; robot.bullet_speed = 25.0;
    InPlateAimState state;
    state.robot_id = target.robot_id; state.armor_id = 0; state.stable_frames = 10;
    state.timestamp = target.timestamp - 10ms;
    PlannerConfig config;
    config.in_plate_aim.enable = true; config.in_plate_aim.mode = InPlateAimMode::Mpc;
    config.in_plate_aim.uncertainty_sigma = 0.0; // 合成滤波初值的协方差未收敛。
    config.enable_air_resistance = drag > 0.0; config.linear_drag_coefficient = drag;
    const auto direct = directPlan(target, robot, config, state);
    Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
    transform.translation() = Eigen::Vector3d{0.02, -0.01, 0.03};
    input.plan = applyTinyMpc(direct, target, robot, transform, config);
    expect(input.plan.using_MPC && input.plan.mpc_aim_feasible,
      "filter snapshot and translated muzzle work in vacuum and drag models");
    expect((input.plan.aim_point_barrel - input.plan.aim_point_world - transform.translation()).norm() < 1e-12,
      "published muzzle translation is consistent");
    input.track_state = L3Estimation::TrackState::Tracking;
    input.plan.fire_permitted = true; input.plan.within_firing_window = true;
    input.actual_yaw = input.plan.yaw; input.actual_pitch = input.plan.pitch;
    L5Control::FireConfig fire;
    fire.shoot_enable = true; fire.stop_fire_on_aim_error = fire.stop_fire_on_temp_lost = false;
    const L5Control::FireDecider decider(fire);
    expect(decider.decide(input).shoot, "verified stationary MPC aim can pass existing fire gates");
    input.plan.mpc_aim_feasible = false;
    const auto transition = decider.decide(input);
    expect(!transition.shoot, "plate-gap feasibility cannot be bypassed by disabled error gate");
    expect(std::find(transition.reasons.begin(), transition.reasons.end(),
      L5Control::RejectReason::MpcAimOutsidePlate) != transition.reasons.end(),
      "transition rejection has a distinct telemetry reason");
    input.track_state = L3Estimation::TrackState::TempLost;
    input.plan.fire_permitted = false;
    expect(!decider.decide(input).shoot, "TempLost continuation cannot bypass plate-gap feasibility");
    input.plan.mpc_aim_feasible = true;
    expect(decider.decide(input).shoot, "feasible TempLost continuation retains configured behavior");
    config.enable_mpc = false;
    const auto disabled = applyTinyMpc(direct, target, robot, transform, config);
    expect(!disabled.using_MPC && disabled.in_plate_aim.offset.norm() == 0.0, "disabled MPC uses center fallback");
  }
}

void stateAndWrapChecks()
{
  L3Estimation::TargetState target;
  target.robot_id = 3; target.center = {4.0, 0.0, 0.2}; target.radius = 0.26;
  target.timestamp = TimePoint{2s}; target.yaw_rate = 0.5; target.yaw = -0.1;
  target.covariance.setZero();
  L1Sensor::RobotState robot;
  robot.timestamp = target.timestamp; robot.bullet_speed = 25.0;
  PlannerConfig config = loadPlannerTuning("config/planner_config.yaml").planner;
  Planner planner(config);
  AimPlan previous;
  for (int i = 0; i < 7; ++i) {
    auto plan = planner.plan(target, robot);
    expect(plan.valid, "fresh observation produces a valid region-mode plan");
    if (i > 0) expect((plan.aim_query_state.offset - previous.in_plate_aim.offset).norm() < 1e-12,
      "selected MPC offset is committed once to next frame snapshot");
    if (i < config.in_plate_aim.warmup_frames)
      expect(plan.in_plate_aim.offset.norm() < 1e-7, "warmup exposes no plate freedom");
    if (i == 2) {
      const auto stale = planner.plan(target, robot);
      expect(stale.aim_query_state.stable_frames == i + 1, "stale frame does not advance observation count");
    }
    previous = plan;
    robot.rpy.yaw = plan.yaw; robot.rpy.pitch = -plan.pitch;
    robot.timestamp += 10ms; target = Predictor{}.predict(target, 0.01);
  }
  expect(previous.using_MPC && previous.in_plate_aim.offset.norm() > 1e-4,
    "runtime 30-percent margin still enables optimized plate freedom after warmup");
  InPlateAimState state;
  state.robot_id = target.robot_id; state.armor_id = 0; state.stable_frames = 10;
  state.timestamp = target.timestamp;
  const auto direct = directPlan(target, robot, config, state);
  const auto base = applyTinyMpc(direct, target, robot, Eigen::Isometry3d::Identity(), config);
  const double turn = std::numbers::pi - 0.005;
  const Eigen::Matrix3d rotation = Eigen::AngleAxisd(turn, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  target.center = rotation * target.center; target.velocity = rotation * target.velocity;
  target.yaw += turn; robot.rpy.yaw = std::remainder(robot.rpy.yaw + turn, 2.0 * std::numbers::pi);
  const auto turned_direct = directPlan(target, robot, config, state);
  const auto turned = applyTinyMpc(turned_direct, target, robot, Eigen::Isometry3d::Identity(), config);
  expect(base.using_MPC && turned.using_MPC
    && std::abs(std::remainder(turned.yaw - base.yaw - turn, 2.0 * std::numbers::pi)) < 1e-6
    && std::abs(turned.pitch - base.pitch) < 1e-6, "region trajectory is consistent across yaw wrap");
}
}  // namespace

int main()
{
  projectionChecks(); optimizerChecks(); pipelineChecks(); candidateSeedChecks(); filterAndFireChecks(); stateAndWrapChecks();
  const auto tuning = loadPlannerTuning("config/planner_config.yaml");
  expect(tuning.planner.in_plate_aim.mode == InPlateAimMode::Mpc, "runtime configuration selects MPC region mode");
  if (!failures) std::cout << "mpc_in_plate_smoke: all checks passed\n";
  return failures ? 1 : 0;
}
