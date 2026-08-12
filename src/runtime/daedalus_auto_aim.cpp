#include "l1_sensor/daedalus_source.hpp"
#include "l1_sensor/serial/robot_state.hpp"
#include "l2_perception/armor.hpp"
#include "l2_perception/armor/armor_detector.hpp"
#include "l2_perception/inference/inference_backend.hpp"
#include "l3_estimation/filter_est/tracker.hpp"
#include "l3_estimation/gtsam_est/tracker.hpp"
#include "l3_estimation/tracker.hpp"
#include "l4_planning/planner.hpp"
#include "l5_control/controller.hpp"
#include "l5_control/fire_decision.hpp"
#include "l6_telemetry/logger.hpp"
#include "l6_telemetry/math.hpp"
#include "runtime/auto_aim_config.hpp"

#include <Eigen/Geometry>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <memory>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

volatile std::sig_atomic_t g_stop_requested = 0;

void requestStop(int) noexcept
{
  g_stop_requested = 1;
}

template<typename T>
[[nodiscard]] T yamlOr(
  const YAML::Node& node,
  const char* key,
  T fallback)
{
  const YAML::Node value = node ? node[key] : YAML::Node{};
  return value ? value.as<T>() : std::move(fallback);
}

[[nodiscard]] double radians(double degrees) noexcept
{
  return degrees * std::numbers::pi / 180.0;
}

[[nodiscard]] L1Sensor::EnemyColor parseEnemyColor(const std::string& value)
{
  if (value == "red") {
    return L1Sensor::EnemyColor::Red;
  }
  if (value == "blue") {
    return L1Sensor::EnemyColor::Blue;
  }
  throw std::runtime_error(
    "robot.enemy_color must be 'red' or 'blue'");
}

[[nodiscard]] L2Perception::ArmorColor armorColor(
  L1Sensor::EnemyColor color) noexcept
{
  switch (color) {
    case L1Sensor::EnemyColor::Red:
      return L2Perception::ArmorColor::Red;
    case L1Sensor::EnemyColor::Blue:
      return L2Perception::ArmorColor::Blue;
    case L1Sensor::EnemyColor::Unknown:
      return L2Perception::ArmorColor::Unknown;
  }
  return L2Perception::ArmorColor::Unknown;
}

struct RuntimeConfig {
  L1Sensor::DaedalusSourceOptions source;
  std::chrono::milliseconds connect_timeout{10000};
  std::chrono::milliseconds poll_sleep{1};

  // 相机/串口以外的自瞄参数全部走共用的 config 加载器，和实车 runtime 同一份
  // schema，避免仿真和实车的参数在两处各写一遍后悄悄分叉。
  runtime::AutoAimConfig auto_aim;
  std::size_t inference_threads{0};

  double bullet_speed{25.0};
  double heat{0.0};
  L1Sensor::EnemyColor enemy_color{L1Sensor::EnemyColor::Blue};

  bool preview{true};
  std::uint64_t log_every_n_frames{60};
  double command_jump_rad{radians(10.0)};
};

[[nodiscard]] RuntimeConfig loadConfig(const std::string& path)
{
  const YAML::Node root = YAML::LoadFile(path);
  RuntimeConfig config;

  const YAML::Node ipc = root["ipc"];
  config.source.meta_path = yamlOr<std::string>(
    ipc, "meta_path", config.source.meta_path);
  config.source.image_pool_path = yamlOr<std::string>(
    ipc, "image_pool_path", config.source.image_pool_path);
  config.source.producer_timeout = std::chrono::milliseconds(
    yamlOr<int>(ipc, "producer_timeout_ms", 1000));
  config.connect_timeout = std::chrono::milliseconds(
    yamlOr<int>(ipc, "connect_timeout_ms", 10000));
  config.poll_sleep = std::chrono::milliseconds(
    yamlOr<int>(ipc, "poll_sleep_ms", 1));

  config.auto_aim = runtime::loadAutoAimConfig(path);

  const YAML::Node inference = root["inference"];
  config.inference_threads = yamlOr<std::size_t>(
    inference, "num_threads", config.inference_threads);

  const YAML::Node robot = root["robot"];
  config.bullet_speed = yamlOr<double>(
    robot, "bullet_speed", config.bullet_speed);
  config.heat = yamlOr<double>(robot, "heat", config.heat);
  config.enemy_color = parseEnemyColor(
    yamlOr<std::string>(robot, "enemy_color", "blue"));

  const YAML::Node runtime_node = root["runtime"];
  config.preview = yamlOr<bool>(runtime_node, "preview", config.preview);
  config.log_every_n_frames = yamlOr<std::uint64_t>(
    runtime_node, "log_every_n_frames", config.log_every_n_frames);
  config.command_jump_rad = radians(
    yamlOr<double>(runtime_node, "command_jump_deg", 10.0));

  if (config.source.producer_timeout <= std::chrono::milliseconds::zero() ||
      config.connect_timeout <= std::chrono::milliseconds::zero() ||
      config.poll_sleep < std::chrono::milliseconds::zero() ||
      !std::isfinite(config.bullet_speed) || config.bullet_speed <= 0.0 ||
      !std::isfinite(config.heat) ||
      !std::isfinite(config.command_jump_rad) ||
      config.command_jump_rad <= 0.0) {
    throw std::runtime_error("daedalus.yaml contains an invalid runtime value");
  }
  if (config.auto_aim.fire.shoot_enable) {
    L6Telemetry::logWarn(
      "simulation firing is ENABLED by config; projectiles may be launched");
  } else {
    L6Telemetry::logInfo(
      "simulation firing is disabled; tracking commands only");
  }
  return config;
}

[[nodiscard]] L2Perception::ArmorDetector makeDetector(
  const RuntimeConfig& config)
{
  auto backend = L2Perception::makeInferenceBackend(config.auto_aim.inference_backend);
  L2Perception::InferenceModelConfig model;
  model.model_path = config.auto_aim.model_path;
  model.device = config.auto_aim.inference_device;
  model.model_color_order = L2Perception::ModelColorOrder::Rgb;
  model.normalization_divisor = 255.0F;
  model.inference_num_threads = config.inference_threads;
  backend->load(model);
  if (!backend->ready()) {
    throw std::runtime_error("inference backend did not become ready");
  }
  L6Telemetry::logInfo(
    "armor model loaded",
    std::string{L2Perception::inferenceBackendName(config.auto_aim.inference_backend)},
    config.auto_aim.model_path.string(), config.auto_aim.inference_device);
  return L2Perception::ArmorDetector(std::move(backend));
}

[[nodiscard]] bool fresh(
  std::chrono::steady_clock::time_point sample,
  std::chrono::steady_clock::time_point now,
  std::chrono::milliseconds maximum_age) noexcept
{
  if (sample.time_since_epoch() ==
      std::chrono::steady_clock::duration::zero()) {
    return false;
  }
  const auto age = now - sample;
  return age >= -std::chrono::milliseconds(5) && age <= maximum_age;
}

void drawDetections(
  cv::Mat& image,
  const std::vector<L2Perception::Armor>& armors)
{
  for (const auto& armor : armors) {
    const cv::Scalar color = armor.color == L2Perception::ArmorColor::Blue
      ? cv::Scalar{255, 0, 0}
      : armor.color == L2Perception::ArmorColor::Red
      ? cv::Scalar{0, 0, 255}
      : cv::Scalar{0, 255, 255};
    for (std::size_t index = 0; index < armor.corners.size(); ++index) {
      cv::line(
        image, armor.corners[index],
        armor.corners[(index + 1) % armor.corners.size()],
        color, 2, cv::LINE_AA);
    }
  }
}

[[nodiscard]] std::unique_ptr<L1Sensor::DaedalusSource> connectSource(
  const RuntimeConfig& config)
{
  const auto deadline =
    std::chrono::steady_clock::now() + config.connect_timeout;
  std::string last_error;
  while (g_stop_requested == 0 &&
         std::chrono::steady_clock::now() < deadline) {
    std::string error;
    auto source = L1Sensor::DaedalusSource::connect(config.source, &error);
    if (source && source->producerAlive()) {
      return source;
    }
    last_error = source ? "Talos heartbeat is stale" : std::move(error);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  throw std::runtime_error(
    "cannot connect to a live Talos simulator: " + last_error);
}

int run(const std::string& config_path)
{
  const RuntimeConfig config = loadConfig(config_path);
  L2Perception::ArmorDetector detector = makeDetector(config);
  auto source = connectSource(config);
  L6Telemetry::logInfo("connected to Talos shared memory");

  std::optional<L1Sensor::DaedalusFrame> first_frame;
  std::optional<L1Sensor::CameraCalibration> calibration;
  const auto first_frame_deadline =
    std::chrono::steady_clock::now() + config.connect_timeout;
  std::string calibration_error;
  while (g_stop_requested == 0 &&
         std::chrono::steady_clock::now() < first_frame_deadline) {
    first_frame = source->read();
    if (first_frame) {
      calibration = source->calibration(*first_frame, &calibration_error);
      if (calibration) {
        break;
      }
    }
    if (!source->producerAlive()) {
      throw std::runtime_error("Talos heartbeat stopped before the first frame");
    }
    std::this_thread::sleep_for(config.poll_sleep);
  }
  if (!first_frame || !calibration) {
    throw std::runtime_error(
      "did not receive a calibrated Talos frame: " + calibration_error);
  }

  L6Telemetry::logInfo(
    "Talos calibration", calibration->image_size.width,
    calibration->image_size.height,
    calibration->camera_matrix.at<double>(0, 0),
    calibration->camera_matrix.at<double>(1, 1));
  std::unique_ptr<L3Estimation::ITracker> tracker;
  if (config.auto_aim.estimator == L3Estimation::EstimatorBackend::Gtsam) {
    tracker = std::make_unique<L3Estimation::GtsamEst::Tracker>(
      *calibration, config.auto_aim.armor, config.auto_aim.target,
      config.auto_aim.gtsam);
  } else {
    tracker = std::make_unique<L3Estimation::FilterEst::Tracker>(
      *calibration, config.auto_aim.armor, config.auto_aim.tracker,
      config.auto_aim.target, config.auto_aim.filter);
  }
  L6Telemetry::logInfo("estimator backend:", L3Estimation::toString(config.auto_aim.estimator));
  if (!tracker->ready()) {
    throw std::runtime_error("Tracker rejected the Talos calibration");
  }

  L4Planning::Planner planner(config.auto_aim.plan);
  const L5Control::FireDecider fire_decider(config.auto_aim.fire);
  const L5Control::Controller controller;
  const L2Perception::ArmorColor expected_color =
    armorColor(config.enemy_color);

  bool preview = config.preview;
  if (preview) {
    try {
      cv::namedWindow("daedalus_auto_aim", cv::WINDOW_NORMAL);
    } catch (const cv::Exception& exception) {
      preview = false;
      L6Telemetry::logWarn("preview disabled", exception.what());
    }
  }

  std::optional<double> last_command_yaw;
  int last_command_armor_id = -1;
  std::uint64_t processed_frames = 0;
  bool following_logged = first_frame->following;
  L6Telemetry::logInfo(
    "Talos F5 auto-aim state", following_logged ? "enabled" : "disabled");

  std::optional<L1Sensor::DaedalusFrame> queued = std::move(first_frame);
  while (g_stop_requested == 0) {
    std::optional<L1Sensor::DaedalusFrame> frame;
    if (queued) {
      frame = std::move(queued);
      queued.reset();
    } else {
      frame = source->read();
    }

    if (!frame) {
      const std::string error = source->takeLastError();
      if (!error.empty()) {
        L6Telemetry::logWarn("Talos frame rejected", error);
      }
      if (!source->producerAlive()) {
        L6Telemetry::logError("Talos heartbeat timed out");
        break;
      }
      std::this_thread::sleep_for(config.poll_sleep);
      continue;
    }

    if (frame->following != following_logged) {
      following_logged = frame->following;
      L6Telemetry::logInfo(
        "Talos F5 auto-aim state",
        following_logged ? "enabled" : "disabled");
    }

    const Eigen::Quaterniond q_world_barrel = frame->gimbal.orientation;
    const Eigen::Vector3d gimbal_ypr =
      L6Telemetry::eulers(q_world_barrel, 2, 1, 0);

    auto armors = detector.detect(frame->bgr_image);
    std::erase_if(armors, [expected_color](const auto& armor) {
      return armor.color != expected_color;
    });

    const auto target = tracker->track(
      armors, q_world_barrel, frame->timestamp);

    L1Sensor::RobotState robot_state;
    robot_state.rpy.yaw = gimbal_ypr[0];
    robot_state.rpy.pitch = gimbal_ypr[1];
    robot_state.rpy.roll = gimbal_ypr[2];
    robot_state.bullet_speed = config.bullet_speed;
    robot_state.heat = config.heat;
    robot_state.enemy_color = config.enemy_color;
    robot_state.mode = L1Sensor::WorkMode::AutoAim;
    robot_state.timestamp = frame->timestamp;

    const auto plan_time = std::chrono::steady_clock::now();
    const L4Planning::Plan plan = planner.plan(
      target, robot_state, plan_time, true);

    L5Control::FireInput fire_input;
    fire_input.target = target;
    fire_input.track_state = tracker->state();
    fire_input.plan = plan;
    fire_input.robot_state = robot_state;
    fire_input.now = plan_time;
    fire_input.actual_yaw = gimbal_ypr[0];
    fire_input.actual_pitch = gimbal_ypr[1];
    fire_input.calibration_ready =
      calibration->barrelExtrinsicsReady();
    fire_input.serial_fresh = frame->following && fresh(
      robot_state.timestamp, plan_time,
      config.auto_aim.fire.max_robot_state_age);
    fire_input.gimbal_pose_fresh = frame->following && fresh(
      frame->gimbal.timestamp, plan_time,
      config.auto_aim.fire.max_gimbal_pose_age);
    fire_input.armor_switching =
      plan.valid && last_command_armor_id >= 0 &&
      plan.armor_id != last_command_armor_id;
    fire_input.command_jump =
      plan.valid && last_command_yaw &&
      std::abs(L6Telemetry::limit_rad(plan.yaw - *last_command_yaw)) >
        config.command_jump_rad;

    const L5Control::FireDecision fire_decision =
      fire_decider.decide(fire_input);
    const auto command = controller.makeCommand(plan, fire_decision);

    bool command_sent = false;
    if (frame->following && command) {
      const double distance = plan.aim_point.norm();
      command_sent = source->sendGimbalCommand(
        command->yaw, command->pitch, distance, command->shoot);
    } else {
      source->sendHold();
    }

    if (command_sent) {
      last_command_yaw = command->yaw;
      last_command_armor_id = plan.armor_id;
    } else if (!plan.valid) {
      last_command_yaw.reset();
      last_command_armor_id = -1;
    }

    ++processed_frames;
    if (config.log_every_n_frames > 0 &&
        processed_frames % config.log_every_n_frames == 0) {
      L6Telemetry::logInfo(
        "Daedalus frame", frame->frame_seq,
        "detections", armors.size(),
        "track_state", static_cast<int>(tracker->state()),
        "plan_valid", plan.valid,
        "following", frame->following,
        "fire_feasible", fire_decision.fire_feasible,
        "shoot", fire_decision.shoot);
    }

    if (preview) {
      drawDetections(frame->bgr_image, armors);
      const std::string status =
        "seq=" + std::to_string(frame->frame_seq) +
        (frame->following ? " F5=on" : " F5=off") +
        (plan.valid ? " plan=valid" : " plan=hold");
      cv::putText(
        frame->bgr_image, status, {20, 40},
        cv::FONT_HERSHEY_SIMPLEX, 0.9,
        {0, 255, 0}, 2, cv::LINE_AA);
      cv::imshow("daedalus_auto_aim", frame->bgr_image);
      const int key = cv::waitKey(1);
      if (key == 27 || key == 'q' || key == 'Q') {
        break;
      }
    }
  }

  source->sendHold();
  if (preview) {
    cv::destroyWindow("daedalus_auto_aim");
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv)
{
  const std::string config_path =
    argc > 1 ? argv[1] : "config/daedalus.yaml";
  std::signal(SIGINT, requestStop);
  std::signal(SIGTERM, requestStop);
  L6Telemetry::initLogger();

  int exit_code = 0;
  try {
    exit_code = run(config_path);
  } catch (const std::exception& exception) {
    L6Telemetry::logError("daedalus_auto_aim stopped", exception.what());
    exit_code = 1;
  }
  L6Telemetry::flushLogger();
  return exit_code;
}
