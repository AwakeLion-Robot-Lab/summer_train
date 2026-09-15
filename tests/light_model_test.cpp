// L2 灯条链路单独回放：灯条关键点模型 → 配对 → 数字分类，逐帧统计和叠加显示。
// 走的是生产的 ArmorDetector，参数全部来自 auto_aim.yaml，只有模型路径和设备
// 允许命令行覆盖；不经过 L3，所以看到的是纯感知层的表现。
//
// 它和 track_diag 的分工：这里回答「灯条检出、配对、数字分类各丢了多少」，
// track_diag 回答「这些观测喂给整车滤波器之后怎么样」。

#include "l2_perception/armor/armor_detector.hpp"
#include "runtime/armor_detector_factory.hpp"
#include "runtime/auto_aim_config.hpp"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using L2Perception::ArmorCandidate;
using L2Perception::ArmorColor;
using L2Perception::Light;
using L2Perception::NumberVerdict;

void require(bool condition, const std::string& message)
{
  if (!condition) {
    throw std::runtime_error(message);
  }
}

const char* verdictName(NumberVerdict verdict)
{
  switch (verdict) {
    case NumberVerdict::Accepted:
      return "ok";
    case NumberVerdict::Negative:
      return "neg";
    case NumberVerdict::LowConfidence:
      return "low";
    case NumberVerdict::TypeMismatch:
      return "type";
  }
  return "?";
}

ArmorColor parseEnemyColor(const std::string& value)
{
  if (value == "red") return ArmorColor::Red;
  if (value == "blue") return ArmorColor::Blue;
  if (value == "any") return ArmorColor::Unknown;
  throw std::invalid_argument("enemy 必须是 red、blue 或 any");
}

double percentile(std::vector<double> values, double ratio)
{
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const std::size_t index = static_cast<std::size_t>(
    std::clamp(ratio, 0.0, 1.0) * static_cast<double>(values.size() - 1));
  return values[index];
}

std::string percent(long long part, long long total)
{
  return total > 0
           ? cv::format("%.1f%%", 100.0 * static_cast<double>(part) / static_cast<double>(total))
           : std::string("-");
}

// 右上角排开每个配对送进 MLP 的数字图，绿框通过、灰框被拒。
void drawNumbers(
  cv::Mat& image, const std::vector<ArmorCandidate>& candidates,
  const L2Perception::NumberClassifier& classifier)
{
  constexpr int kScale = 3;
  constexpr int kCellWidth = 20 * kScale;
  constexpr int kCellHeight = 28 * kScale;
  constexpr int kLabelHeight = 16;
  int x = image.cols - kCellWidth - 8;
  const int y = 44;
  for (const ArmorCandidate& candidate : candidates) {
    if (x < 0 || candidate.number.number_image.empty() ||
        y + kCellHeight + kLabelHeight > image.rows) {
      break;
    }
    cv::Mat cell;
    cv::resize(candidate.number.number_image, cell, {kCellWidth, kCellHeight}, 0.0, 0.0,
               cv::INTER_NEAREST);
    cv::cvtColor(cell, cell, cv::COLOR_GRAY2BGR);
    cell.copyTo(image(cv::Rect(x, y, kCellWidth, kCellHeight)));
    const cv::Scalar border = candidate.number.verdict == NumberVerdict::Accepted
                                ? cv::Scalar{0, 255, 0}
                                : cv::Scalar{128, 128, 128};
    cv::rectangle(image, cv::Rect(x, y, kCellWidth, kCellHeight), border, 1);
    cv::putText(image, classifier.label(candidate.number.label_index),
                {x + 2, y + kCellHeight + kLabelHeight - 4}, cv::FONT_HERSHEY_SIMPLEX, 0.45,
                border, 1, cv::LINE_AA);
    x -= kCellWidth + 6;
  }
}

void draw(cv::Mat& image, const std::vector<Light>& lights,
          const std::vector<ArmorCandidate>& candidates,
          const L2Perception::NumberClassifier& classifier, const std::string& header)
{
  cv::putText(image, header, {12, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {0, 255, 0}, 2,
              cv::LINE_AA);

  for (const Light& light : lights) {
    const cv::Scalar color = light.color == ArmorColor::Red ? cv::Scalar{0, 0, 255}
                                                            : cv::Scalar{255, 0, 0};
    cv::line(image, light.top, light.bottom, color, 1, cv::LINE_AA);
    cv::circle(image, light.top, 3, {0, 255, 0}, -1, cv::LINE_AA);
    cv::circle(image, light.bottom, 3, {255, 0, 255}, -1, cv::LINE_AA);
    cv::putText(image, cv::format("%.2f/%.0fpx", light.score, light.length),
                light.top + cv::Point2f{6.0F, -4.0F}, cv::FONT_HERSHEY_SIMPLEX, 0.4,
                {200, 255, 200}, 1, cv::LINE_AA);
  }

  // 被拒的配对先画、细灰线；通过的后画、绿色粗线，重叠时通过的在上面。
  for (const bool accepted_pass : {false, true}) {
    for (const ArmorCandidate& candidate : candidates) {
      const bool accepted = candidate.number.verdict == NumberVerdict::Accepted;
      if (accepted != accepted_pass) {
        continue;
      }
      const Light& left = lights[candidate.pair.left];
      const Light& right = lights[candidate.pair.right];
      const cv::Scalar color = accepted ? cv::Scalar{0, 255, 0} : cv::Scalar{128, 128, 128};
      cv::line(image, left.top, right.bottom, color, accepted ? 2 : 1, cv::LINE_AA);
      cv::line(image, left.bottom, right.top, color, accepted ? 2 : 1, cv::LINE_AA);
      const std::string& label = classifier.label(candidate.number.label_index);
      const double confidence = candidate.number.confidence * 100.0;
      // negative 本身就是标签，不再重复写原因；其余被拒的写「原因:标签」。
      const std::string text =
        accepted || candidate.number.verdict == NumberVerdict::Negative
          ? cv::format("%s %.0f%%", label.c_str(), confidence)
          : cv::format("%s:%s %.0f%%", verdictName(candidate.number.verdict), label.c_str(),
                       confidence);
      cv::putText(image, text, left.top + cv::Point2f{0.0F, -18.0F}, cv::FONT_HERSHEY_SIMPLEX,
                  accepted ? 0.6 : 0.4, accepted ? cv::Scalar{0, 255, 255} : color,
                  accepted ? 2 : 1, cv::LINE_AA);
    }
  }

  drawNumbers(image, candidates, classifier);
}

}  // namespace

int main(int argc, char** argv)
{
  // cv::CommandLineParser 只认 -c=value 形式；空格分隔会把值吃成位置参数。
  const cv::String keys =
    "{help h        |                     | 显示帮助}"
    "{@input        | records/3m_run_fast | 录像路径，可省略 .avi 后缀}"
    "{model         |                     | 灯条关键点模型，留空用 auto_aim.yaml 的}"
    "{device        | CPU                 | OpenVINO 设备}"
    "{enemy         | blue                | 只配对该颜色的灯条：red / blue / any}"
    "{wait          | 1                   | imshow 等待毫秒，0 为逐帧}"
    "{start-index   | 0                   | 起始帧}"
    "{end-index     | -1                  | 结束帧，-1 到末尾}"
    "{save          |                     | 写可视化帧到该目录，留空不写}"
    "{csv           |                     | 每帧统计写到该 csv，留空不写}"
    "{headless      | false               | 不开窗口，配合 --save 用}";

  cv::CommandLineParser cli(argc, argv, keys);
  cli.about("L2 灯条链路回放");
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }

  try {
    std::filesystem::path input = cli.get<cv::String>("@input");
    if (input.extension().empty()) {
      input += ".avi";
    }
    require(std::filesystem::exists(input), "找不到录像 " + input.string());

    // 检测器与实机同一个工厂组装，只有模型路径和设备允许命令行覆盖。
    runtime::AutoAimConfig config = runtime::loadConfig("config/auto_aim.yaml");
    const std::string model_override = cli.get<cv::String>("model");
    if (!model_override.empty()) {
      config.inference.model_path = model_override;
    }
    config.inference.device = cli.get<cv::String>("device");
    const L2Perception::ArmorDetector detector = runtime::makeDetector(config);
    std::cout << "灯条模型 " << config.inference.model_path << "  数字模型 "
              << config.number_classifier.model_path << '\n';

    const ArmorColor enemy_color = parseEnemyColor(cli.get<cv::String>("enemy"));
    const int start_index = cli.get<int>("start-index");
    const int end_index = cli.get<int>("end-index");
    const int wait_ms = cli.get<int>("wait");
    const bool headless = cli.get<bool>("headless");
    const std::string save_dir = cli.get<cv::String>("save");
    const std::string csv_path = cli.get<cv::String>("csv");
    if (!save_dir.empty()) {
      std::filesystem::create_directories(save_dir);
    }

    cv::VideoCapture capture(input.string());
    require(capture.isOpened(), "打不开录像 " + input.string());
    if (start_index > 0) {
      capture.set(cv::CAP_PROP_POS_FRAMES, start_index);
    }

    std::ofstream csv;
    if (!csv_path.empty()) {
      csv.open(csv_path);
      csv << "frame,lights,pairs,accepted,negative,low_conf,type_mismatch,l2_ms,labels\n";
    }

    std::vector<double> l2_times;
    std::vector<double> scores;
    std::vector<double> lengths;
    std::vector<double> tilts;
    long long light_total = 0;
    long long empty_light_frames = 0;
    long long pair_total = 0;
    long long frames_with_armor = 0;
    std::map<NumberVerdict, long long> verdict_counts;
    std::map<std::string, long long> accepted_labels;

    cv::Mat frame;
    int frame_index = start_index;
    std::size_t frames = 0;
    while (capture.read(frame)) {
      if (end_index >= 0 && frame_index > end_index) {
        break;
      }

      const auto begin = std::chrono::steady_clock::now();
      const L2Perception::ArmorFrame perception =
        detector.detectFrame(frame, std::nullopt, std::nullopt, enemy_color);
      const double l2_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin)
          .count();
      l2_times.push_back(l2_ms);
      ++frames;

      const std::vector<Light>& lights = detector.lastLights();
      const std::vector<ArmorCandidate>& candidates = detector.lastCandidates();
      light_total += static_cast<long long>(lights.size());
      empty_light_frames += lights.empty() ? 1 : 0;
      for (const Light& light : lights) {
        scores.push_back(light.score);
        lengths.push_back(light.length);
        tilts.push_back(light.tilt_angle_deg);
      }
      pair_total += static_cast<long long>(candidates.size());
      frames_with_armor += perception.armors.empty() ? 0 : 1;

      std::map<NumberVerdict, int> frame_verdicts;
      std::string frame_labels;
      for (const ArmorCandidate& candidate : candidates) {
        ++frame_verdicts[candidate.number.verdict];
        ++verdict_counts[candidate.number.verdict];
        if (candidate.number.verdict == NumberVerdict::Accepted) {
          const std::string& label =
            detector.classifier().label(candidate.number.label_index);
          ++accepted_labels[label];
          frame_labels += (frame_labels.empty() ? "" : " ") + label;
        }
      }

      if (csv.is_open()) {
        csv << frame_index << ',' << lights.size() << ',' << candidates.size() << ','
            << frame_verdicts[NumberVerdict::Accepted] << ','
            << frame_verdicts[NumberVerdict::Negative] << ','
            << frame_verdicts[NumberVerdict::LowConfidence] << ','
            << frame_verdicts[NumberVerdict::TypeMismatch] << ',' << l2_ms << ','
            << frame_labels << '\n';
      }

      if (!headless || !save_dir.empty()) {
        cv::Mat canvas = frame.clone();
        draw(canvas, lights, candidates, detector.classifier(),
             cv::format("frame %d | %zu lights | %zu pairs, %zu armors | %.1f ms", frame_index,
                        lights.size(), candidates.size(), perception.armors.size(), l2_ms));
        if (!save_dir.empty()) {
          cv::imwrite(save_dir + cv::format("/light_%05d.jpg", frame_index), canvas);
        }
        if (!headless) {
          cv::imshow("light pipeline", canvas);
          const int key = cv::waitKey(wait_ms);
          if (key == 27 || key == 'q') {
            break;
          }
        }
      }
      ++frame_index;
    }
    require(frames > 0, "一帧都没读到");

    const long long accepted = verdict_counts[NumberVerdict::Accepted];
    const auto per_frame = [frames](long long count) {
      return cv::format("%.2f", static_cast<double>(count) / static_cast<double>(frames));
    };
    std::cout << "\n帧数 " << frames << '\n'
              << "灯条     " << light_total << " 根  每帧 " << per_frame(light_total)
              << "  零检出帧 " << empty_light_frames << '\n'
              << "  分数 p10 " << cv::format("%.2f", percentile(scores, 0.10)) << "  p50 "
              << cv::format("%.2f", percentile(scores, 0.50)) << '\n'
              << "  长度 p10 " << cv::format("%.1f", percentile(lengths, 0.10)) << "  p50 "
              << cv::format("%.1f", percentile(lengths, 0.50)) << "  p90 "
              << cv::format("%.1f", percentile(lengths, 0.90)) << " px\n"
              << "  倾角 p50 " << cv::format("%.1f", percentile(tilts, 0.50)) << "  p90 "
              << cv::format("%.1f", percentile(tilts, 0.90)) << " deg\n"
              << "配对     " << pair_total << " 对  每帧 " << per_frame(pair_total) << '\n'
              << "  通过 " << accepted << " (" << percent(accepted, pair_total) << ")  negative "
              << verdict_counts[NumberVerdict::Negative] << "  低置信 "
              << verdict_counts[NumberVerdict::LowConfidence] << "  板型不符 "
              << verdict_counts[NumberVerdict::TypeMismatch] << '\n'
              << "  有装甲板的帧 " << frames_with_armor << " ("
              << percent(frames_with_armor, static_cast<long long>(frames)) << ")\n"
              << "  类别";
    for (const auto& [label, count] : accepted_labels) {
      std::cout << "  " << label << ':' << count;
    }
    std::cout << "\nL2 整帧 p50 " << cv::format("%.1f", percentile(l2_times, 0.50))
              << " ms  p90 " << cv::format("%.1f", percentile(l2_times, 0.90)) << " ms\n"
              << "light model test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "light model test failed: " << error.what() << '\n';
    return 1;
  }
}
