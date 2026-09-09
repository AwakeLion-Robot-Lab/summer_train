#include "daedalus_probe_camera.hpp"
#include "l2_perception/armor/armor_detector.hpp"
#include "l3_estimation/armor/tracker.hpp"
#include "l4_planning/armor/planner.hpp"
#include "l6_telemetry/aim_overlay.hpp"
#include "l6_telemetry/logger.hpp"
#include "l6_telemetry/quintic_trace.hpp"
#include "l6_telemetry/udp_json_sender.hpp"
#include "runtime/auto_aim_config.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <atomic>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <sstream>
#include <thread>

namespace {
using Clock = std::chrono::steady_clock;
using Json = nlohmann::json;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
std::atomic<bool> running{true};
void stop(int) { running = false; }

struct Options {
  std::string config{"config/auto_aim.yaml"};
  std::string csv;
  std::string save;
  std::string host{"127.0.0.1"};
  int port{9870};
  L1Sensor::DaedalusPaths paths;
  bool synthetic{false}, realtime{false}, show{false}, follow{false};
  bool udp{true}, blend{true};
  double duration{0.0}, hz{200.0}, omega{6.0}, distance{4.0}, radius{0.26};
  double yaw_acc{0.0}, pitch_acc{0.0}, yaw_speed{10.0}, pitch_speed{10.0};
  double bullet_speed{25.0};
  std::size_t frames{0};
  L2Perception::ArmorColor enemy{L2Perception::ArmorColor::Unknown};
};

double number(const std::string& value)
{
  std::size_t end = 0;
  const double result = std::stod(value, &end);
  if (end != value.size() || !std::isfinite(result)) {
    throw std::invalid_argument("invalid numeric option: " + value);
  }
  return result;
}

Options parse(int argc, char** argv)
{
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string arg{argv[i]};
    const auto equal = arg.find('=');
    const auto key = arg.substr(0, equal);
    const auto value = equal == std::string::npos ? "" : arg.substr(equal + 1);
    if (arg == "--help" || arg == "-h") {
      std::cout <<
        "Usage: daedalus_quintic_probe [options]\n"
        "  --synthetic               deterministic rotating target, no simulator/model\n"
        "  --realtime                pace synthetic samples for live PlotJuggler\n"
        "  --duration=S --frames=N   stop condition (0 = unlimited; synthetic default 8s)\n"
        "  --hz=200 --omega=6 --distance=4 --radius=0.26  synthetic target\n"
        "  --config=PATH             existing auto_aim YAML (model and blend settings)\n"
        "  --csv=PATH                save all curves; default logs/quintic_TIMESTAMP.csv\n"
        "  --host=127.0.0.1 --port=9870 --no-udp\n"
        "  --show --save=PATH         live image overlay / final PNG\n"
        "  --follow                  send planned angles to simulator; fire is always off\n"
        "  --enemy=any|red|blue       live detector color\n"
        "  --no-blend                compare unmodified shooting trajectory\n"
        "  --yaw-acc=50 --pitch-acc=100  override polynomial acceleration limits (rad/s^2)\n"
        "  --yaw-speed=10 --pitch-speed=10  diagnostic speed thresholds (rad/s)\n"
        "  --bullet-speed=25         match simulator projectile speed (m/s)\n"
        "  --metadata=PATH --image-pool=PATH  Talos IPC files\n\n"
        "Live uses nv-merge's existing detector/Tracker/Planner. Synthetic bypasses L2/L3.\n"
        "PlotJuggler: UDP Server, JSON, port 9870, timestamp field t.\n";
      std::exit(0);
    } else if (arg == "--synthetic") o.synthetic = true;
    else if (arg == "--realtime") o.realtime = true;
    else if (arg == "--show") o.show = true;
    else if (arg == "--follow") o.follow = true;
    else if (arg == "--no-udp") o.udp = false;
    else if (arg == "--no-blend") o.blend = false;
    else if (!value.empty()) {
      if (key == "--config") o.config = value;
      else if (key == "--csv") o.csv = value;
      else if (key == "--save") o.save = value;
      else if (key == "--host") o.host = value;
      else if (key == "--metadata") o.paths.metadata = value;
      else if (key == "--image-pool") o.paths.image_pool = value;
      else if (key == "--enemy") {
        if (value == "red") o.enemy = L2Perception::ArmorColor::Red;
        else if (value == "blue") o.enemy = L2Perception::ArmorColor::Blue;
        else if (value != "any") throw std::invalid_argument("invalid --enemy");
      } else {
        const double v = number(value);
        if (key == "--duration") o.duration = v;
        else if (key == "--hz") o.hz = v;
        else if (key == "--omega") o.omega = v;
        else if (key == "--distance") o.distance = v;
        else if (key == "--radius") o.radius = v;
        else if (key == "--bullet-speed") o.bullet_speed = v;
        else if (key == "--yaw-speed") o.yaw_speed = v;
        else if (key == "--pitch-speed") o.pitch_speed = v;
        else if (key == "--yaw-acc" && v > 0.0) o.yaw_acc = v;
        else if (key == "--pitch-acc" && v > 0.0) o.pitch_acc = v;
        else if (key == "--port" && v >= 1 && v <= 65535 && std::floor(v) == v)
          o.port = static_cast<int>(v);
        else if (key == "--frames" && v >= 0 && v <= 1e9 && std::floor(v) == v)
          o.frames = static_cast<std::size_t>(v);
        else throw std::invalid_argument("unknown or invalid option: " + arg);
      }
    } else throw std::invalid_argument("unknown or incomplete option: " + arg);
  }
  if (o.duration < 0 || o.hz < 1 || o.hz > 2000 || o.distance <= o.radius ||
      o.radius <= 0 || o.yaw_speed <= 0 || o.pitch_speed <= 0 || o.bullet_speed < 14) {
    throw std::invalid_argument("invalid duration, sampling rate, geometry or limits");
  }
  if (o.synthetic && (o.follow || o.show || !o.save.empty())) {
    throw std::invalid_argument("--follow/--show/--save require live simulator input");
  }
  if (o.synthetic && o.duration == 0 && o.frames == 0) o.duration = 8.0;
  if (o.csv.empty()) {
    const auto stamp = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count();
    o.csv = "logs/quintic_" + std::to_string(stamp) + ".csv";
  }
  return o;
}

class Recorder {
public:
  explicit Recorder(const Options& o)
  {
    const auto parent = std::filesystem::path(o.csv).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    file_.open(o.csv);
    if (!file_) throw std::runtime_error("cannot write CSV: " + o.csv);
    file_ << std::setprecision(17);
    if (o.udp) sender_ = std::make_unique<L6Telemetry::UdpJsonSender>(o.host, o.port);
  }
  void write(const Json& packet)
  {
    const Json flat = packet.flatten();
    if (keys_.empty()) {
      keys_.push_back("/t");
      for (const auto& [key, value] : flat.items()) {
        (void)value;
        if (key != "/t") keys_.push_back(key);
      }
      file_ << "t";
      for (std::size_t i = 1; i < keys_.size(); ++i) file_ << ',' << keys_[i];
      file_ << '\n';
    }
    if (flat.size() != keys_.size()) throw std::runtime_error("telemetry schema changed");
    for (std::size_t i = 0; i < keys_.size(); ++i) {
      if (i) file_ << ',';
      const auto& value = flat.at(keys_[i]);
      if (value.is_number() && std::isfinite(value.get<double>())) file_ << value;
    }
    file_ << '\n';
    if (!file_) throw std::runtime_error("CSV write failed");
    if (sender_ && !sender_->send(packet)) ++udp_errors;
    if (++rows_ % 100 == 0) file_.flush();
  }
  std::size_t udp_errors{0};
private:
  std::ofstream file_;
  std::vector<std::string> keys_;
  std::unique_ptr<L6Telemetry::UdpJsonSender> sender_;
  std::size_t rows_{0};
};

void overlay(cv::Mat& image, const L4Planning::Plan& plan, const Json& data,
             const L1Sensor::CameraCalibration& calibration, const Eigen::Quaterniond& pose)
{
  std::ostringstream status;
  status << "Quintic probe | valid=" << plan.valid() << " blend=" << plan.aim.blending
         << " commits=" << data["counts"]["commits"]
         << " acc-infeasible=" << data["counts"]["acc_infeasible_commits"];
  L6Telemetry::drawOutlinedText(image, status.str(), {12, 30}, {0, 255, 255}, 0.55);
  L6Telemetry::drawOutlinedText(image, "orange=shoot reference | green=planned | fire=OFF",
    {12, 56}, {255, 255, 255}, 0.5);
  if (!plan.valid()) return;
  const auto marker = [&](double yaw, double pitch, cv::Scalar color) {
    const Eigen::Vector3d ray{
      std::cos(pitch) * std::cos(yaw), std::cos(pitch) * std::sin(yaw), -std::sin(pitch)};
    if (const auto pixel = L6Telemetry::projectWorldPoint(ray * 5.0, calibration, pose)) {
      cv::drawMarker(image, *pixel, color, cv::MARKER_CROSS, 20, 2);
    }
  };
  marker(plan.aim.shootYaw(), plan.aim.shootPitch(), {0, 165, 255});
  marker(plan.aim.yaw, plan.aim.pitch, {0, 255, 0});
}
}  // namespace

int main(int argc, char** argv)
{
  try {
    const auto o = parse(argc, argv);
    std::signal(SIGINT, stop);
    std::signal(SIGTERM, stop);
    L6Telemetry::initLogger();
    auto config = runtime::loadAutoAimConfig(o.config);
    // 测试入口内覆盖仿真参数，不改用户现场 YAML。
    config.plan.blend.enable = o.blend;
    config.plan.impact.send_to_control = 0.0;
    config.plan.impact.high_speed_delay_time = 0.0;
    config.plan.impact.low_speed_delay_time = 0.0;
    config.plan.ballistic.gravity = 9.81;
    config.plan.ballistic.drag_coefficient = 0.0;
    if (o.yaw_acc > 0) config.plan.blend.limits.max_yaw_acceleration = o.yaw_acc;
    if (o.pitch_acc > 0) config.plan.blend.limits.max_pitch_acceleration = o.pitch_acc;
    L4Planning::Planner planner(config.plan);
    planner.enableDiagnostics(true);
    L6Telemetry::QuinticTrace trace(config.plan.blend.limits, o.yaw_speed, o.pitch_speed);
    Recorder recorder(o);
    std::unique_ptr<L2Perception::ArmorDetector> detector;
    std::unique_ptr<L3Estimation::Tracker> tracker;
    std::optional<L1Sensor::CameraCalibration> calibration;
    std::unique_ptr<L3Estimation::PnpSolver> overlay_solver;
    L1Sensor::DaedalusClient client(o.paths);
    if (!o.synthetic) {
      const auto deadline = Clock::now() + std::chrono::seconds{5};
      while (!client.connect() || !client.isSimulatorAlive()) {
        if (!running || Clock::now() >= deadline)
          throw std::runtime_error("cannot connect to live Daedalus: " + client.lastError());
        std::this_thread::sleep_for(std::chrono::milliseconds{100});
      }
      auto backend = L2Perception::makeInferenceBackend(config.inference_backend);
      backend->load(config.inference);
      detector = std::make_unique<L2Perception::ArmorDetector>(
        std::move(backend), config.decoder, L2Perception::ImagePreprocessConfig{}, config.refiner);
      if (!detector->ready()) throw std::runtime_error("L2 not ready");
      if (o.show) cv::namedWindow("Daedalus quintic probe", cv::WINDOW_NORMAL);
    }
    const auto wall_start = Clock::now();
    const auto synthetic_origin = L4Planning::TimePoint{};
    L3Estimation::TrackedTarget synthetic_target(
      L3Estimation::ArmorName::Infantry3, o.distance, o.omega, o.radius, 0.0,
      L3Estimation::HeightOffsets{0.05, 0.0, 0.0});
    synthetic_target.jumped = true;
    std::uint64_t tracker_resets = 0;
    std::size_t count = 0, valid_count = 0;
    double measured_send_delay = 0.0;
    bool camera_ready_before = false;
    std::vector<double> planning_times;
    cv::Mat last_image;
    Json last;
    std::cout << "Source=" << (o.synthetic ? "synthetic" : "Daedalus images")
      << " | CSV=" << o.csv << " | UDP=" << (o.udp ? std::to_string(o.port) : "off")
      << " | blend=" << o.blend << " | follow=" << o.follow << " | fire=OFF\n";
    while (running && (o.frames == 0 || count < o.frames)) {
      const double synthetic_t = static_cast<double>(count) / o.hz;
      const double wall_t = std::chrono::duration<double>(Clock::now() - wall_start).count();
      if (o.duration > 0 && (o.synthetic ? synthetic_t : wall_t) >= o.duration) break;
      if (o.synthetic && o.realtime) {
        std::this_thread::sleep_until(wall_start + std::chrono::duration_cast<Clock::duration>(
          std::chrono::duration<double>{synthetic_t}));
      }
      L4Planning::PlanInput input;
      input.robot_state.bullet_speed = o.bullet_speed;
      input.plan_to_send = measured_send_delay;
      double measured_yaw = kNaN, measured_pitch = kNaN;
      L1Sensor::DaedalusFrame frame;
      std::optional<Eigen::Quaterniond> pose;
      std::size_t detections = 0;
      double perception_us = 0.0;
      bool camera_ready = false;
      if (o.synthetic) {
        auto target = synthetic_target;
        input.plan_time = synthetic_origin + std::chrono::duration_cast<Clock::duration>(
          std::chrono::duration<double>{synthetic_t});
        target.predict(input.plan_time);
        input.target = std::move(target);
      } else {
        if (!client.readFrame(frame, std::chrono::seconds{1})) {
          // 不跨采集断层计算导数，也不继续沿用旧目标。
          planner.reset();
          trace.reset();
          if (tracker) tracker->reset();
          if (!client.isSimulatorAlive() || !client.lastError().empty())
            throw std::runtime_error("Daedalus frame stream stopped: " + client.lastError());
          continue;
        }
        pose = DaedalusProbe::barrelPose(frame);
        if (!pose) throw std::runtime_error("invalid simulator barrel quaternion");
        const Eigen::Vector3d forward = *pose * Eigen::Vector3d::UnitX();
        measured_yaw = std::atan2(forward.y(), forward.x());
        measured_pitch = -std::atan2(forward.z(), std::hypot(forward.x(), forward.y()));
        const auto perception_start = Clock::now();
        camera_ready = DaedalusProbe::cameraNearBarrel(frame);
        if (camera_ready) {
          // 必须使用本帧的相机信息。启动远景或切视角前的 Camera 平移可能相差
          // 十几米，缓存第一帧会把目标变到枪管后方，造成 yaw 翻转和 EKF 发散。
          calibration = DaedalusProbe::makeCalibration(frame);
          if (!tracker) {
            tracker = std::make_unique<L3Estimation::Tracker>(
              *calibration, config.armor, config.tracker, config.target);
            overlay_solver = std::make_unique<L3Estimation::PnpSolver>(*calibration, config.armor);
          } else if (!tracker->setCalibration(*calibration) ||
                     !overlay_solver->setCalibration(*calibration)) {
            throw std::runtime_error("invalid current-frame simulator calibration");
          }
          if (!tracker->ready()) throw std::runtime_error("L3 not ready");
          auto armors = detector->detect(frame.image_bgr);
          if (o.enemy != L2Perception::ArmorColor::Unknown)
            std::erase_if(armors, [&](const auto& a) { return a.color != o.enemy; });
          detections = armors.size();
          input.target = tracker->track(armors, pose, frame.capture_time);
        } else if (camera_ready_before || count == 0) {
          if (tracker) tracker->reset();
          std::cout << "Waiting for Robot camera view: camera is not near the barrel.\n";
        }
        camera_ready_before = camera_ready;
        if (tracker && tracker->resetCount() != tracker_resets) {
          planner.reset();
          trace.reset();
          tracker_resets = tracker->resetCount();
        }
        perception_us = std::chrono::duration<double, std::micro>(
          Clock::now() - perception_start).count();
        input.plan_time = Clock::now();
      }
      const auto plan_start = Clock::now();
      const auto plan = planner.plan(input);
      const auto plan_end = Clock::now();
      const double plan_us = std::chrono::duration<double, std::micro>(plan_end - plan_start).count();
      planning_times.push_back(plan_us);
      if (plan.valid()) ++valid_count;
      bool command_sent = false;
      if (!o.synthetic && o.follow && plan.valid() && frame.auto_aim_enabled) {
        constexpr double degrees = 180.0 / std::numbers::pi;
        const float distance = plan.fire ? static_cast<float>(plan.fire->point().norm()) : 1.0F;
        command_sent = client.sendGimbalCommand(
          static_cast<float>(plan.aim.yaw * degrees),
          static_cast<float>(plan.aim.pitch * degrees), distance, false);
        if (!command_sent) throw std::runtime_error("simulator command failed: " + client.lastError());
      }
      measured_send_delay = command_sent
        ? std::chrono::duration<double>(Clock::now() - plan_end).count() : 0.0;
      last = trace.update(input.plan_time, plan, planner.diagnostics(), measured_yaw, measured_pitch);
      last["timing"] = {{"plan_us", plan_us}, {"perception_us", perception_us},
        {"image_age_ms", o.synthetic ? 0.0
          : std::chrono::duration<double, std::milli>(input.plan_time - frame.capture_time).count()}};
      last["source"] = {{"synthetic", int(o.synthetic)}, {"frame", o.synthetic ? count : frame.sequence},
        {"detections", detections}, {"tracker_state", tracker ? static_cast<int>(tracker->state()) : -1},
        {"tracker_resets", tracker_resets}, {"auto_aim_enabled", int(frame.auto_aim_enabled)},
        {"camera_ready", int(camera_ready)},
        {"command_sent", int(command_sent)}, {"blend_enabled", int(o.blend)},
        {"target_omega_rad_s", input.target ? input.target->ekf_x()[7] : kNaN}};
      const auto& camera_position = frame.pose(L1Sensor::DaedalusPoseKind::Camera).position;
      last["camera"] = {
        {"x_barrel_m", o.synthetic ? kNaN : camera_position[0]},
        {"y_barrel_m", o.synthetic ? kNaN : camera_position[1]},
        {"z_barrel_m", o.synthetic ? kNaN : camera_position[2]}};
      recorder.write(last);
      ++count;
      if (!o.synthetic && (o.show || !o.save.empty())) {
        if (overlay_solver) overlay_solver->set_R_world_barrel(pose);
        if (input.target && overlay_solver) {
          L6Telemetry::drawVehicle(frame.image_bgr, input.target->armor_xyza_list(),
            input.target->armor_type, input.target->name, *overlay_solver, {0, 255, 0}, 2);
        }
        if (calibration) overlay(frame.image_bgr, plan, last, *calibration, *pose);
        last_image = frame.image_bgr;
        if (o.show) {
          cv::Mat display;
          cv::resize(last_image, display, {}, 0.7, 0.7);
          cv::imshow("Daedalus quintic probe", display);
          const int key = cv::waitKey(1);
          if (key == 'q' || key == 27) break;
        }
      }
      if (count == 1 || count % 200 == 0) {
        std::cout << "frames=" << count << " valid=" << valid_count
          << " commits=" << last["counts"]["commits"]
          << " infeasible=" << last["counts"]["acc_infeasible_commits"]
          << " blend_fraction=" << last["coverage"]["blend_fraction"] << '\n';
      }
    }
    if (!o.save.empty() && (last_image.empty() || !cv::imwrite(o.save, last_image)))
      throw std::runtime_error("failed to save overlay: " + o.save);
    if (o.show) cv::destroyWindow("Daedalus quintic probe");
    std::sort(planning_times.begin(), planning_times.end());
    const auto percentile = [&](double q) {
      return planning_times.empty() ? 0.0 : planning_times[static_cast<std::size_t>(
        q * static_cast<double>(planning_times.size() - 1))];
    };
    std::cout << "Done: frames=" << count << " valid=" << valid_count
      << " plan_us p50=" << percentile(0.5) << " p95=" << percentile(0.95)
      << " p99=" << percentile(0.99) << " UDP errors=" << recorder.udp_errors
      << "\nCSV: " << o.csv << '\n';
    return count == 0 ? 2 : 0;
  } catch (const std::exception& error) {
    std::cerr << "quintic probe: " << error.what() << '\n';
    return 1;
  }
}
