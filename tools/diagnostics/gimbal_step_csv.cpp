#include "l1_sensor/serial/serial_port.hpp"
#include "l1_sensor/serial/serial_protocol.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numbers>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using State = L1Sensor::RobotState;
constexpr double kRadToDeg = 180.0 / std::numbers::pi;
constexpr double kDegToRad = std::numbers::pi / 180.0;
constexpr auto kTxPeriod = std::chrono::milliseconds(5);
volatile std::sig_atomic_t interrupted = 0;

void onSignal(int) { interrupted = 1; }

struct Options {
  std::string port = "/dev/ttyACM0";
  int baud = 115200;
  std::string axis;
  std::vector<double> initials;
  std::vector<double> deltas;
  double min_deg = 0.0;
  double max_deg = 0.0;
  bool has_limits = false;
  std::string output;
  int repeats = 1;
  int duration_ms = 2000;
  int settle_timeout_ms = 5000;
  std::optional<double> onset_threshold_deg;
  bool arm = false;
};

double number(const std::string& value)
{
  std::size_t consumed = 0;
  const double parsed = std::stod(value, &consumed);
  if (consumed != value.size() || !std::isfinite(parsed)) {
    throw std::invalid_argument("invalid number: " + value);
  }
  return parsed;
}

int positiveInteger(const std::string& value)
{
  const double parsed = number(value);
  if (parsed < 1 || parsed > 1000000 || std::floor(parsed) != parsed) {
    throw std::invalid_argument("expected positive integer: " + value);
  }
  return static_cast<int>(parsed);
}

std::vector<double> list(const std::string& value)
{
  std::vector<double> result;
  std::size_t begin = 0;
  while (begin <= value.size()) {
    const auto end = value.find(',', begin);
    const auto part = value.substr(begin, end == std::string::npos ? end : end - begin);
    if (part.empty()) throw std::invalid_argument("empty angle in list");
    result.push_back(number(part));
    if (end == std::string::npos) break;
    begin = end + 1;
  }
  return result;
}

void usage(const char* program)
{
  std::cerr << "单组角度：\n  " << program
            << " --axis yaw|pitch --initial-deg 0 --offset-deg 10"
               " --repeat 5 --limits -45:45 --output /tmp/gimbal_yaw [--arm]\n"
            << "多组角度：\n  " << program
            << " --axis yaw|pitch --initials -30,0,30 --offsets -10,10"
               " --repeat 5 --limits -45:45 --output /tmp/gimbal_grid [--arm]\n"
            << "--initial-deg  起始绝对角（度）；--offset-deg  相对偏转角（度）。\n"
            << "--repeat       每组初始角和偏转角重复次数，默认 1。\n"
            << "--limits       该轴允许的最小:最大角度（度），必须填写。\n"
            << "--duration-ms  每次命令后的观测窗口，正整数 ms，默认 2000 ms。\n"
            << "--onset-threshold-deg  固定响应角阈值（正数）；省略时使用 max(0.5°, 5%步长)。\n"
            << "--port /dev/ttyACM0  --baud 115200  --settle-timeout-ms 5000\n"
            << "不加 --arm 只预览，不打开串口、不移动云台。\n";
}

Options parse(int argc, char* argv[])
{
  Options options;
  std::optional<double> single_initial;
  std::optional<double> single_offset;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    if (key == "--") continue;
    std::optional<std::string> inline_value;
    if (const auto equal = key.find('='); equal != std::string::npos) {
      inline_value = key.substr(equal + 1);
      key.resize(equal);
    }
    if (key == "--arm") {
      if (inline_value) throw std::invalid_argument("--arm does not take a value");
      options.arm = true;
      continue;
    }
    if (key == "--help" || key == "-h") { usage(argv[0]); std::exit(0); }
    if (!inline_value && i + 1 >= argc)
      throw std::invalid_argument("missing value for " + key);
    const std::string value = inline_value ? *inline_value : argv[++i];
    if (key == "--port") options.port = value;
    else if (key == "--baud") options.baud = positiveInteger(value);
    else if (key == "--axis") options.axis = value;
    else if (key == "--initial-deg") single_initial = number(value);
    else if (key == "--offset-deg") single_offset = number(value);
    else if (key == "--initials") options.initials = list(value);
    else if (key == "--offsets" || key == "--deltas") options.deltas = list(value);
    else if (key == "--output") options.output = value;
    else if (key == "--repeat" || key == "--repeats") options.repeats = positiveInteger(value);
    else if (key == "--duration-ms") options.duration_ms = positiveInteger(value);
    else if (key == "--onset-threshold-deg") {
      options.onset_threshold_deg = number(value);
      if (*options.onset_threshold_deg <= 0.0)
        throw std::invalid_argument("--onset-threshold-deg must be positive");
    }
    else if (key == "--settle-timeout-ms") options.settle_timeout_ms = positiveInteger(value);
    else if (key == "--limits") {
      const auto colon = value.find(':');
      if (colon == std::string::npos) throw std::invalid_argument("limits must be min:max");
      options.min_deg = number(value.substr(0, colon));
      options.max_deg = number(value.substr(colon + 1));
      options.has_limits = true;
    } else throw std::invalid_argument("unknown option: " + key);
  }
  if (single_initial.has_value() != single_offset.has_value()) {
    throw std::invalid_argument("--initial-deg and --offset-deg must be given together");
  }
  if (single_initial && (!options.initials.empty() || !options.deltas.empty())) {
    throw std::invalid_argument("single-angle and multi-angle options cannot be mixed");
  }
  if (single_initial) {
    options.initials = {*single_initial};
    options.deltas = {*single_offset};
  }
  if ((options.axis != "yaw" && options.axis != "pitch") || options.initials.empty() ||
      options.deltas.empty() || !options.has_limits || options.output.empty() ||
      options.min_deg >= options.max_deg) {
    throw std::invalid_argument("axis, angle(s), limits, and output are required");
  }
  for (double initial : options.initials) {
    for (double delta : options.deltas) {
      if (delta == 0.0) throw std::invalid_argument("offset angle cannot be zero");
      if (initial < options.min_deg || initial > options.max_deg ||
          initial + delta < options.min_deg || initial + delta > options.max_deg)
        throw std::invalid_argument("initial " + std::to_string(initial) +
                                    " + offset " + std::to_string(delta) +
                                    " = target " + std::to_string(initial + delta) +
                                    " is outside limits [" + std::to_string(options.min_deg) +
                                    ", " + std::to_string(options.max_deg) + "]");
    }
  }
  return options;
}

double angleDeg(const State& state, const std::string& axis)
{
  return (axis == "yaw" ? state.rpy.yaw : state.rpy.pitch) * kRadToDeg;
}

double difference(double a, double b, const std::string& axis)
{
  return axis == "yaw" ? std::remainder(a - b, 360.0) : a - b;
}

struct Trial {
  int id;
  int repeat;
  double initial;
  double delta;
};

struct Sample {
  double t_ms;
  double yaw_deg;
  double pitch_deg;
  bool baseline;
};

struct Metrics {
  double initial_actual;
  double onset_ms;
  double rise_ms;
  double settling_ms;
  double overshoot_deg;
  double final_error_deg;
  int sample_count;
};

struct CurveTrial {
  double initial_actual;
  std::vector<std::pair<double, double>> progress;
};

double missing() { return std::numeric_limits<double>::quiet_NaN(); }

double median(std::vector<double> values)
{
  values.erase(std::remove_if(values.begin(), values.end(),
                              [](double v) { return !std::isfinite(v); }), values.end());
  if (values.empty()) return missing();
  std::sort(values.begin(), values.end());
  const auto mid = values.size() / 2;
  return values.size() % 2 ? values[mid] : (values[mid - 1] + values[mid]) / 2.0;
}

double interpolatedProgress(const CurveTrial& trial, double t_ms)
{
  const auto upper = std::lower_bound(
    trial.progress.begin(), trial.progress.end(), t_ms,
    [](const auto& point, double time) { return point.first < time; });
  if (upper == trial.progress.end()) return missing();
  if (upper == trial.progress.begin()) return upper->second;
  const auto lower = upper - 1;
  const double span = upper->first - lower->first;
  if (span <= 0.0) return upper->second;
  const double alpha = (t_ms - lower->first) / span;
  return lower->second + alpha * (upper->second - lower->second);
}

class Rig {
public:
  explicit Rig(const Options& options) : axis_(options.axis), port_([&] {
    L1Sensor::SerialConfig config;
    config.device = options.port;
    config.baud_rate = options.baud;
    config.read_timeout_ms = 2;
    return config;
  }()) {}

  void open()
  {
    if (!port_.open()) throw std::runtime_error("cannot open serial port");
  }

  std::optional<State> poll()
  {
    std::uint8_t byte{};
    if (port_.read(std::span<std::uint8_t>{&byte, 1}) != 1) return std::nullopt;
    const auto frames = protocol_.feed(std::span<const std::uint8_t>{&byte, 1});
    if (frames.empty()) return std::nullopt;
    const auto state = frames.back();
    if (!std::isfinite(state.rpy.yaw) || !std::isfinite(state.rpy.pitch)) {
      throw std::runtime_error("nonfinite angle in status frame");
    }
    if (state.mode != L1Sensor::WorkMode::AutoAim) {
      throw std::runtime_error("lower controller left AutoAim mode");
    }
    last_ = state;
    return state;
  }

  State waitForState()
  {
    const auto deadline = Clock::now() + std::chrono::seconds(3);
    while (Clock::now() < deadline && !interrupted) {
      if (auto state = poll()) return *state;
    }
    throw std::runtime_error("no AutoAim status frame received within 3 seconds");
  }

  Clock::time_point command(double active_deg, double fixed_deg)
  {
    L5Control::SerialCommand command;
    command.yaw = (axis_ == "yaw" ? active_deg : fixed_deg) * kDegToRad;
    // command.pitch = (axis_ == "pitch" ? active_deg : fixed_deg) * kDegToRad;
    command.pitch = 0;
    command.shoot = false;
    const auto bytes = protocol_.encodeCommand(command);
    const auto before_write = Clock::now();
    if (port_.write(std::span<const std::uint8_t>{bytes.data(), bytes.size()}) != bytes.size()) {
      throw std::runtime_error("serial command write failed");
    }
    return before_write;
  }

  template <class Fn>
  void drive(double active, double fixed, Clock::time_point end, Fn onState)
  {
    auto next_tx = Clock::now();
    while (Clock::now() < end && !interrupted) {
      const auto now = Clock::now();
      if (now >= next_tx) {
        command(active, fixed);
        next_tx += kTxPeriod;
        if (next_tx < now) next_tx = now + kTxPeriod;
      }
      if (auto state = poll()) onState(*state);
    }
    if (interrupted) throw std::runtime_error("interrupted");
  }

  void settle(double initial, double fixed, int timeout_ms)
  {
    std::optional<Clock::time_point> stable_since;
    auto next_tx = Clock::now();
    const auto deadline = next_tx + std::chrono::milliseconds(timeout_ms);
    while (Clock::now() < deadline && !interrupted) {
      const auto now = Clock::now();
      if (now >= next_tx) {
        command(initial, fixed);
        next_tx += kTxPeriod;
        if (next_tx < now) next_tx = now + kTxPeriod;
      }
      if (auto state = poll()) {
        if (std::abs(difference(angleDeg(*state, axis_), initial, axis_)) <= 1.0) {
          if (!stable_since) stable_since = state->timestamp;
          if (state->timestamp - *stable_since >= std::chrono::milliseconds(300)) return;
        } else stable_since.reset();
      }
    }
    throw std::runtime_error(interrupted ? "interrupted" : "initial angle did not settle within timeout");
  }

  const State& last() const { return *last_; }

private:
  std::string axis_;
  L1Sensor::SerialPort port_;
  L1Sensor::SerialProtocol protocol_;
  std::optional<State> last_;
};

Metrics metrics(const std::vector<Sample>& samples, double initial_actual,
                double delta, double target, const std::string& axis,
                double onset_threshold_deg)
{
  Metrics result{initial_actual, missing(), missing(), missing(), 0.0, missing(), 0};
  double t10 = missing();
  double t90 = missing();
  std::vector<double> final_angles;
  std::vector<double> errors;
  double last_t = missing();
  const double settle_band = std::max(0.2, std::abs(delta) * 0.02);
  for (const auto& sample : samples) {
    if (sample.baseline) continue;
    ++result.sample_count;
    const double actual = axis == "yaw" ? sample.yaw_deg : sample.pitch_deg;
    const double travel = difference(actual, initial_actual, axis);
    const double progress = travel / delta;
    if (!std::isfinite(result.onset_ms) && travel * (delta > 0.0 ? 1.0 : -1.0) >=
        onset_threshold_deg)
      result.onset_ms = sample.t_ms;
    if (!std::isfinite(t10) && progress >= 0.1) t10 = sample.t_ms;
    if (!std::isfinite(t90) && progress >= 0.9) t90 = sample.t_ms;
    result.overshoot_deg = std::max(result.overshoot_deg, (progress - 1.0) * std::abs(delta));
    errors.push_back(std::abs(difference(actual, target, axis)));
    last_t = sample.t_ms;
    final_angles.push_back(actual);
  }
  if (std::isfinite(t10) && std::isfinite(t90)) result.rise_ms = t90 - t10;
  if (!final_angles.empty()) {
    const std::size_t tail = std::min<std::size_t>(final_angles.size(), 5);
    std::vector<double> final_errors;
    for (auto it = final_angles.end() - static_cast<std::ptrdiff_t>(tail);
         it != final_angles.end(); ++it)
      final_errors.push_back(difference(target, *it, axis));
    result.final_error_deg = median(final_errors);
  }
  std::size_t index = 0;
  for (const auto& sample : samples) {
    if (sample.baseline) continue;
    if (last_t - sample.t_ms >= 100.0 && errors[index] <= settle_band &&
        std::all_of(errors.begin() + static_cast<std::ptrdiff_t>(index), errors.end(),
                    [settle_band](double error) { return error <= settle_band; })) {
      result.settling_ms = sample.t_ms;
      break;
    }
    ++index;
  }
  return result;
}

void csvNumber(std::ostream& output, double value)
{
  if (std::isfinite(value)) output << value;
}

int run(const Options& options)
{
  std::vector<Trial> trials;
  for (int repeat = 1; repeat <= options.repeats; ++repeat)
    for (double initial : options.initials)
      for (double delta : options.deltas)
        trials.push_back({0, repeat, initial, delta});
  std::mt19937 generator{20261021};
  std::shuffle(trials.begin(), trials.end(), generator);
  for (std::size_t i = 0; i < trials.size(); ++i) trials[i].id = static_cast<int>(i + 1);
  std::cout << "测试轴=" << options.axis << "，每组重复=" << options.repeats
            << " 次，总次数=" << trials.size() << "，每次记录="
            << options.duration_ms << " ms\n"
            << "初始角(度)  偏转角(度)  目标角(度)  重复次数\n";
  for (double initial : options.initials)
    for (double delta : options.deltas)
      std::cout << std::setw(12) << initial << std::setw(14) << delta
                << std::setw(14) << initial + delta << std::setw(10)
                << options.repeats << '\n';
  if (!options.arm) {
    std::cout << "仅预览。确认后加 --arm 才会打开串口、移动云台并写 CSV。\n";
    return 0;
  }

  Rig rig(options);
  rig.open();
  const State first = rig.waitForState();
  std::cout << "AutoAim status received; current " << options.axis << '='
            << angleDeg(first, options.axis) << " deg\n";

  std::ofstream raw(options.output + "_samples.csv");
  std::ofstream summaries(options.output + "_trials.csv");
  std::ofstream model(options.output + "_model.csv");
  std::ofstream curve(options.output + "_curve.csv");
  if (!raw || !summaries || !model || !curve)
    throw std::runtime_error("cannot create CSV output files");
  raw << "trial_id,repeat,axis,initial_cmd_deg,initial_actual_deg,delta_cmd_deg,target_deg,phase,t_ms,command_yaw_deg,command_pitch_deg,actual_yaw_deg,actual_pitch_deg\n";
  summaries << "trial_id,repeat,axis,initial_cmd_deg,initial_actual_deg,delta_cmd_deg,target_deg,sample_count,onset_ms,rise_10_90_ms,settling_ms,overshoot_deg,final_error_deg\n";
  model << "axis,initial_cmd_deg,delta_cmd_deg,trial_count,median_onset_ms,median_rise_10_90_ms,median_settling_ms,median_overshoot_deg,median_final_error_deg\n";
  curve << "axis,initial_cmd_deg,delta_cmd_deg,t_ms,trial_count,median_progress,median_actual_deg\n";
  raw << std::fixed << std::setprecision(6);
  summaries << std::fixed << std::setprecision(6);
  model << std::fixed << std::setprecision(6);
  curve << std::fixed << std::setprecision(6);
  std::map<std::pair<double, double>, std::vector<Metrics>> groups;
  std::map<std::pair<double, double>, std::vector<CurveTrial>> curves;

  for (const auto& trial : trials) {
    const double fixed = angleDeg(rig.last(), options.axis == "yaw" ? "pitch" : "yaw");
    std::cout << "试验 " << trial.id << '/' << trials.size() << "：初始角="
              << trial.initial << "°，偏转=" << trial.delta << "°，目标角="
              << trial.initial + trial.delta << "°，第 " << trial.repeat
              << " 次重复" << std::endl;
    rig.settle(trial.initial, fixed, options.settle_timeout_ms);
    std::vector<std::pair<Clock::time_point, State>> before;
    rig.drive(trial.initial, fixed, Clock::now() + std::chrono::milliseconds(500),
              [&](const State& state) { before.emplace_back(state.timestamp, state); });
    if (before.size() < 5) throw std::runtime_error("too few baseline status frames");
    std::vector<double> baseline_angles;
    for (const auto& [timestamp, state] : before)
      baseline_angles.push_back(trial.initial + difference(angleDeg(state, options.axis), trial.initial, options.axis));
    const double initial_actual = median(baseline_angles);
    const double target = trial.initial + trial.delta;
    const auto t0 = rig.command(target, fixed);
    std::vector<Sample> samples;
    samples.reserve(before.size() + static_cast<std::size_t>(options.duration_ms / 5));
    for (const auto& [timestamp, state] : before)
      samples.push_back({std::chrono::duration<double, std::milli>(timestamp - t0).count(),
                         state.rpy.yaw * kRadToDeg, state.rpy.pitch * kRadToDeg, true});
    rig.drive(target, fixed, t0 + std::chrono::milliseconds(options.duration_ms),
              [&](const State& state) {
                samples.push_back({std::chrono::duration<double, std::milli>(state.timestamp - t0).count(),
                                   state.rpy.yaw * kRadToDeg, state.rpy.pitch * kRadToDeg, false});
              });
    const double onset_threshold = options.onset_threshold_deg.value_or(
      std::max(0.5, std::abs(trial.delta) * 0.05));
    const Metrics m = metrics(samples, initial_actual, trial.delta, target, options.axis,
                              onset_threshold);
    std::cout << "  响应启动延迟：";
    if (std::isfinite(m.onset_ms)) {
      std::cout << m.onset_ms << " ms";
    } else {
      std::cout << "未测到（" << options.duration_ms << " ms 窗口内"
                << (m.sample_count == 0 ? "没有状态反馈" : "未达到响应阈值") << "）";
    }
    std::cout << "，阈值=" << onset_threshold
              << "°（沿命令方向），反馈样本=" << m.sample_count << '\n';
    for (const auto& sample : samples) {
      const double cmd_yaw = options.axis == "yaw" ? (sample.baseline ? trial.initial : target) : fixed;
      const double cmd_pitch = options.axis == "pitch" ? (sample.baseline ? trial.initial : target) : fixed;
      raw << trial.id << ',' << trial.repeat << ',' << options.axis << ',' << trial.initial
          << ',' << initial_actual << ',' << trial.delta << ',' << target << ','
          << (sample.baseline ? "baseline" : "step") << ',' << sample.t_ms << ','
          << cmd_yaw << ',' << cmd_pitch << ',' << sample.yaw_deg << ',' << sample.pitch_deg << '\n';
    }
    summaries << trial.id << ',' << trial.repeat << ',' << options.axis << ',' << trial.initial
              << ',' << initial_actual << ',' << trial.delta << ',' << target << ',' << m.sample_count << ',';
    csvNumber(summaries, m.onset_ms); summaries << ',';
    csvNumber(summaries, m.rise_ms); summaries << ',';
    csvNumber(summaries, m.settling_ms); summaries << ',';
    csvNumber(summaries, m.overshoot_deg); summaries << ',';
    csvNumber(summaries, m.final_error_deg); summaries << '\n';
    raw.flush();
    summaries.flush();
    groups[{trial.initial, trial.delta}].push_back(m);
    CurveTrial response{initial_actual, {{0.0, 0.0}}};
    for (const auto& sample : samples) {
      if (sample.baseline || sample.t_ms <= response.progress.back().first) continue;
      const double actual = options.axis == "yaw" ? sample.yaw_deg : sample.pitch_deg;
      response.progress.emplace_back(
        sample.t_ms, difference(actual, initial_actual, options.axis) / trial.delta);
    }
    curves[{trial.initial, trial.delta}].push_back(std::move(response));
  }

  for (const auto& [condition, values] : groups) {
    std::array<std::vector<double>, 5> columns;
    for (const auto& value : values) {
      columns[0].push_back(value.onset_ms);
      columns[1].push_back(value.rise_ms);
      columns[2].push_back(value.settling_ms);
      columns[3].push_back(value.overshoot_deg);
      columns[4].push_back(value.final_error_deg);
    }
    model << options.axis << ',' << condition.first << ',' << condition.second << ',' << values.size();
    for (auto& column : columns) { model << ','; csvNumber(model, median(column)); }
    model << '\n';
  }
  for (const auto& [condition, responses] : curves) {
    for (int t = 0; t < options.duration_ms; t += 10) {
      std::vector<double> progress;
      std::vector<double> actual;
      for (const auto& response : responses) {
        const double value = interpolatedProgress(response, static_cast<double>(t));
        if (!std::isfinite(value)) continue;
        progress.push_back(value);
        actual.push_back(response.initial_actual + value * condition.second);
      }
      if (progress.empty()) continue;
      curve << options.axis << ',' << condition.first << ',' << condition.second << ','
            << t << ',' << progress.size() << ',' << median(progress) << ','
            << median(actual) << '\n';
    }
  }
  std::cout << "Wrote " << options.output << "_{samples,trials,model,curve}.csv\n";
  return 0;
}

}  // namespace

int main(int argc, char* argv[])
{
  std::signal(SIGINT, onSignal);
  std::signal(SIGTERM, onSignal);
  try { return run(parse(argc, argv)); }
  catch (const std::exception& error) {
    std::cerr << "gimbal_step_csv: " << error.what() << '\n';
    usage(argv[0]);
    return 1;
  }
}
