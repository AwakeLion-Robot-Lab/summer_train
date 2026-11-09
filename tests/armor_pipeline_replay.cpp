// Offline A/B of the production vision stage and bounded handoff. Uses fixed
// image/pose/planning timestamps to check output equivalence across scheduling.
// No USB, serial I/O, imshow or frame dropping; this is NOT end-to-end latency.
#define main autoAimExecutableMain
#include "../src/runtime/auto_aim_runtime.cpp"
#undef main

#include <bit>
#include <iostream>
#include <numeric>

namespace {
struct ReplayPose {
  std::optional<Eigen::Quaterniond> roiPosePreview(Clock::time_point) const { return std::nullopt; }
  bool waitPoseUntil(Clock::time_point, Clock::time_point) const { return true; }
  std::optional<Eigen::Quaterniond> gimbalPoseAt(Clock::time_point) const {
    return Eigen::Quaterniond::Identity();
  }
};

struct ReplaySummary {
  std::uint64_t hash{14695981039346656037ULL};
  int targets{0}, valid_plans{0}, detections{0};
};

ReplaySummary replay(const std::vector<cv::Mat>& images, double fps,
                     const runtime::AutoAimConfig& config,
                     const L1Sensor::CameraCalibration& calibration,
                     const L4Planning::PlannerTuning& tuning, bool pipelined) {
  auto detector = runtime::makeDetector(config);
  for (int i = 0; i < 30; ++i) detector.detect(images[i % images.size()]);
  std::optional<L3Estimation::EskfTracker> tracker;
  tracker.emplace(calibration, config.armor, config.ieskf_tracker, config.ieskf_target);
  L4Planning::Planner planner(tuning.planner);
  L4Planning::PlannerContext context;
  context.config = tuning.planner;
  context.latency = tuning.latency;
  context.armor_score_weights = tuning.armor_score_weights;
  context.facing_angle_good = tuning.facing_angle_good;
  context.facing_angle_bad = tuning.facing_angle_bad;
  L5Control::Controller controller(config.fire, config.runtime.command_jump_threshold);
  L3Estimation::PnpSolver overlay_solver(calibration, config.armor);
  L1Sensor::SerialConfig serial_config;
  serial_config.enable = false;
  L1Sensor::SerialWorker serial(serial_config); // Only read telemetry counters.
  const ReplayPose poses;
  int index = 0;
  const int count = static_cast<int>(images.size());
  const auto produce = [&]() -> std::optional<ArmorVisionPacket> {
    if (index == count) return std::nullopt;
    ArmorVisionPacket packet;
    packet.image = images[index].clone(); // Consumer draws; never mutate replay inputs.
    packet.timing.captured_at = Clock::time_point(std::chrono::seconds(1000)) +
      std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(index / fps));
    packet.timing.ready_at = Clock::now();
    packet.state = L1Sensor::RobotState{};
    packet.state->enemy_color = L1Sensor::EnemyColor::Blue;
    packet.state->bullet_speed = tuning.default_bullet_speed;
    packet.state->timestamp = packet.timing.captured_at;
    processArmorVision(packet, detector, tracker, poses, true, 20);
    ++index;
    return packet;
  };
  ReplaySummary result;
  const auto mix = [&](std::uint64_t v) { result.hash ^= v; result.hash *= 1099511628211ULL; };
  std::vector<double> waits, consumers, visions;
  const auto consume = [&](ArmorVisionPacket packet) {
    const auto started = Clock::now();
    waits.push_back(milliseconds(started, packet.completed_at));
    visions.push_back(packet.timing.detect_ms + packet.timing.estimate_ms + packet.timing.pose_wait_ms);
    auto state = *packet.state;
    context.planning_time = packet.timing.captured_at + std::chrono::milliseconds(10);
    state.timestamp = context.planning_time;
    const auto plan = planner.plan(runtime::toL4TargetState(packet.target), state, context);
    const auto command = controller.update(packet.target, packet.track_state, plan,
                                          Eigen::Quaterniond::Identity(), true);
    mix(static_cast<unsigned>(packet.track_state));
    mix(packet.match_count);
    mix(packet.perception.armors.size());
    result.detections += packet.perception.armors.size();
    for (const auto& armor : packet.perception.armors) {
      mix(armor.class_id); mix(static_cast<unsigned>(armor.color));
      mix(std::bit_cast<std::uint32_t>(armor.confidence));
      for (const auto& p : armor.corners) {
        mix(std::bit_cast<std::uint32_t>(p.x)); mix(std::bit_cast<std::uint32_t>(p.y));
      }
    }
    if (packet.target) {
      ++result.targets;
      for (const double value : packet.target->rawState()) mix(std::bit_cast<std::uint64_t>(value));
    }
    mix(plan.valid);
    if (plan.valid) {
      ++result.valid_plans;
      mix(std::bit_cast<std::uint64_t>(plan.yaw));
      mix(std::bit_cast<std::uint64_t>(plan.pitch));
    }
    mix(command.has_value());
    if (command) {
      mix(command->shoot);
      mix(std::bit_cast<std::uint64_t>(static_cast<double>(command->yaw)));
      mix(std::bit_cast<std::uint64_t>(static_cast<double>(command->pitch)));
    }
    // Real telemetry serialization and overlay drawing, without network or GUI.
    const auto telemetry = telemetryFrame(packet.image_pose, state, {}, packet.target,
      packet.track_state, plan, controller.lastDecision(), command.has_value(), serial,
      packet.timing.captured_at, packet.match_count, packet.tracker_drops, 0).dump();
    if (telemetry.empty()) throw std::runtime_error("empty telemetry");
    overlay_solver.set_R_world_barrel(packet.image_pose);
    L6Telemetry::drawAimOverlay(packet.image,
      {.detections = packet.perception.armors, .observations = {}, .target = packet.target,
       .track_state = packet.track_state, .plan = plan, .fire = controller.lastDecision(),
       .q_world_barrel = packet.image_pose}, overlay_solver, calibration);
    consumers.push_back(milliseconds(Clock::now(), started));
  };
  const auto started = Clock::now();
  if (pipelined) {
    runtime::SingleFramePipeline<ArmorVisionPacket> pipe(produce);
    for (int i = 0; i < count; ++i) {
      ArmorVisionPacket packet;
      if (pipe.readFor(packet, std::chrono::seconds(10)) != decltype(pipe)::ReadStatus::Value)
        throw std::runtime_error("replay handoff timed out");
      consume(std::move(packet));
    }
  } else {
    while (auto packet = produce()) consume(std::move(*packet));
  }
  const double elapsed = milliseconds(Clock::now(), started);
  const auto mean = [](const auto& values) {
    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
  };
  std::sort(waits.begin(), waits.end());
  std::cout << (pipelined ? "pipeline" : "serial") << " fps=" << count * 1000.0 / elapsed
    << " cycle_ms=" << elapsed / count << " vision_ms=" << mean(visions)
    << " consumer_ms=" << mean(consumers) << " wait_p95_ms=" << waits[waits.size() * 95 / 100]
    << " targets=" << result.targets << " valid_plans=" << result.valid_plans
    << " detections=" << result.detections << " hash=" << result.hash << '\n';
  if (!result.targets || !result.valid_plans) throw std::runtime_error("replay did not exercise tracking/planning");
  return result;
}
} // namespace

int main(int argc, char** argv) {
  try {
    auto config = runtime::loadAutoAimConfig("config/auto_aim.yaml");
    if (argc > 2) config.inference.device = argv[2];
    const auto tuning = L4Planning::loadPlannerTuning("config/planner_config.yaml");
    const auto calibration = L1Sensor::loadCameraCalibration(
      YAML::LoadFile("config/camera_config.yaml")["calibration"], "replay");
    cv::VideoCapture video(argc > 1 ? argv[1] : "video/3m_run_mid.avi");
    double fps = video.get(cv::CAP_PROP_FPS);
    if (!std::isfinite(fps) || fps <= 0) throw std::runtime_error("invalid video fps");
    std::vector<cv::Mat> images;
    for (int i = 0; i < 240; ++i) {
      cv::Mat frame;
      if (!video.read(frame)) break;
      images.push_back(std::move(frame));
    }
    if (images.empty()) throw std::runtime_error("no replay frames");
    std::optional<std::uint64_t> expected;
    for (const bool pipelined : {false, true, true, false}) {
      const auto result = replay(images, fps, config, calibration, tuning, pipelined);
      if (expected && result.hash != *expected) throw std::runtime_error("serial/pipeline output mismatch");
      expected = result.hash;
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
