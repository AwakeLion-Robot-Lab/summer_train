// 板 ROI 内传统角点精修的离线回放工具：逐帧跑生产的 L2 检测器，把每块装甲板
// 的精修 ROI 框出来并标注判定结果，没精修成的（保留网络角点）用黄框标出、写明
// 原因。调 refiner.binary_threshold 时唯一可靠的手段是肉眼确认端点贴不贴灯条。
//
//   xmake run armor_refiner_video_test
//   xmake run armor_refiner_video_test -- records/3m_run_fast.avi
//   xmake run armor_refiner_video_test -- --wait=0          # 逐帧手动推进
//   xmake run armor_refiner_video_test -- --only-kept=true  # 只停在有板没精修成的帧
//   xmake run armor_refiner_video_test -- --headless=true   # 无显示器，只出统计
#include "l2_perception/armor/armor_detector.hpp"
#include "runtime/armor_detector_factory.hpp"
#include "runtime/auto_aim_config.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

namespace
{

const std::string kCommandLineKeys =
  "{help h usage ? | false | 输出命令行参数说明}"
  "{model m |  | 整板模型，留空用 auto_aim.yaml 的；给了就按输出形状认 layout}"
  "{device d | CPU | OpenVINO 推理设备}"
  "{wait w | 30 | 每帧 waitKey 毫秒，0 表示逐帧手动推进}"
  "{start-index s | 0 | 视频起始帧下标}"
  "{end-index e | 0 | 视频结束帧下标，0 表示到结尾}"
  "{only-kept | false | 只在有板保留网络角点的帧暂停}"
  "{patch | true | 是否显示各 ROI 的二值化结果窗口}"
  "{threshold t | -1 | 覆盖 refiner.binary_threshold，<0 表示用 YAML 的}"
  "{save-kept | | 把有板保留网络角点的帧存到该目录，留空表示不存}"
  "{headless | false | 不开窗口，仅统计与存图，便于无显示器环境批量跑}"
  "{@input | records/3m_high.avi | 输入视频路径}";

// BGR。判定与颜色的对应关系在图例里同步显示，避免看图时靠记忆。
const cv::Scalar kRefinedColor{80, 220, 80};         // 绿：已精修
const cv::Scalar kKeptColor{60, 200, 235};           // 黄：保留网络角点
const cv::Scalar kNetworkCornerColor{220, 160, 60};  // 蓝：网络原始角点
const cv::Scalar kTextColor{240, 240, 240};

cv::Scalar verdictColor(L2Perception::RefineVerdict verdict)
{
  return verdict == L2Perception::RefineVerdict::Refined ? kRefinedColor : kKeptColor;
}

const char* verdictName(L2Perception::RefineVerdict verdict)
{
  return verdict == L2Perception::RefineVerdict::Refined ? "REFINED" : "KEPT";
}

// 判定落到该结果的原因。没精修成时最需要看的就是这一行。
std::string verdictReason(const L2Perception::RefineRecord& record)
{
  if (record.verdict == L2Perception::RefineVerdict::Refined) {
    return cv::format("L+R found, shift %.1fpx", record.corner_shift);
  }
  if (record.out_of_image) {
    return "roi leaves the image, skipped";
  }
  if (record.size_skipped) {
    return cv::format("lightbar %.1fpx < min, skipped", record.lightbar_length);
  }
  if (record.shift_rejected) {
    return cv::format("endpoint distance %.1fpx too large", record.corner_shift);
  }
  return cv::format("fewer than two bars, aspect %.2f", record.aspect_ratio);
}

void drawQuad(
  cv::Mat& image, const std::array<cv::Point2f, 4>& corners, const cv::Scalar& color,
  int thickness)
{
  for (std::size_t index = 0; index < corners.size(); ++index) {
    cv::line(
      image, corners[index], corners[(index + 1) % corners.size()], color, thickness,
      cv::LINE_AA);
  }
}

// 把各 ROI 的二值化结果拼成一条横向面板。调 threshold 时看这个窗口最直接：
// 灯条是否完整、是否粘连、光晕有没有被圈进来，一眼就能判断。
cv::Mat buildPatchPanel(
  const cv::Mat& frame, const std::vector<L2Perception::RefineRecord>& records,
  double threshold)
{
  constexpr int kPatchHeight = 160;
  constexpr int kGap = 8;
  std::vector<cv::Mat> tiles;
  const cv::Rect image_rect(0, 0, frame.cols, frame.rows);

  for (const auto& record : records) {
    const cv::Rect roi = record.roi & image_rect;
    if (roi.width < 3 || roi.height < 3) {
      continue;
    }
    cv::Mat gray;
    cv::cvtColor(frame(roi), gray, cv::COLOR_BGR2GRAY);
    cv::Mat binary;
    cv::threshold(gray, binary, threshold, 255.0, cv::THRESH_BINARY);

    cv::Mat colored;
    cv::cvtColor(binary, colored, cv::COLOR_GRAY2BGR);
    const double scale = static_cast<double>(kPatchHeight) / roi.height;
    cv::Mat resized;
    cv::resize(colored, resized, {}, scale, scale, cv::INTER_NEAREST);
    // 给每块贴上判定色边框，和主画面里的 ROI 一一对应。
    cv::rectangle(
      resized, cv::Rect(0, 0, resized.cols, resized.rows), verdictColor(record.verdict), 2);
    tiles.push_back(resized);
  }

  if (tiles.empty()) {
    return {};
  }

  int total_width = 0;
  int max_height = 0;
  for (const cv::Mat& tile : tiles) {
    total_width += tile.cols + kGap;
    max_height = std::max(max_height, tile.rows);
  }

  cv::Mat panel(max_height, total_width, CV_8UC3, cv::Scalar::all(30));
  int x = 0;
  for (const cv::Mat& tile : tiles) {
    tile.copyTo(panel(cv::Rect(x, 0, tile.cols, tile.rows)));
    x += tile.cols + kGap;
  }
  return panel;
}

}  // namespace

int main(int argc, char** argv)
{
  cv::CommandLineParser cli(argc, argv, kCommandLineKeys);
  if (cli.get<bool>("help")) {
    cli.printMessage();
    return 0;
  }

  const std::string input = cli.get<std::string>("@input");
  const std::string model = cli.get<std::string>("model");
  const std::string device = cli.get<std::string>("device");
  const int wait_ms = cli.get<int>("wait");
  const int start_index = cli.get<int>("start-index");
  const int end_index = cli.get<int>("end-index");
  const bool only_kept = cli.get<bool>("only-kept");
  const bool show_patch = cli.get<bool>("patch");
  const double threshold_override = cli.get<double>("threshold");
  const std::string save_kept = cli.get<std::string>("save-kept");
  const bool headless = cli.get<bool>("headless");
  if (!cli.check()) {
    cli.printErrors();
    return 1;
  }

  cv::VideoCapture video(input);
  if (!video.isOpened()) {
    std::printf("failed to open video: %s\n", input.c_str());
    return 1;
  }

  // 参数从 auto_aim.yaml 读，回放和实机用同一份；--threshold 只是临时覆盖，
  // 方便扫阈值，扫出来的值要写回 YAML 才对实机生效。只看精修，侧边灯条用
  // 传统一路，免得为这个工具多加载一个模型。
  runtime::AutoAimConfig config = runtime::loadConfig("config/auto_aim.yaml");
  if (threshold_override >= 0.0) {
    config.refiner.binary_threshold = threshold_override;
  }
  config.inference.device = device;
  const bool model_overridden = !model.empty();
  if (model_overridden) {
    config.inference.model_path = model;
  }

  L2Perception::ArmorDetector detector;
  try {
    detector = runtime::makeDetector(config, model_overridden);
  } catch (const std::exception& error) {
    std::printf(
      "failed to load model %s: %s\n", config.inference.model_path.string().c_str(),
      error.what());
    return 1;
  }
  // 只有离线工具需要逐块明细，实机路径保持关闭。
  detector.collectRecords(true);

  if (start_index > 0) {
    video.set(cv::CAP_PROP_POS_FRAMES, start_index);
  }

  std::printf(
    "input=%s  model=%s  device=%s  threshold=%.0f\n", input.c_str(),
    config.inference.model_path.string().c_str(), device.c_str(),
    config.refiner.binary_threshold);
  if (!headless) {
    std::printf("keys: space=pause/step  q/ESC=quit\n");
  }

  L2Perception::RefineStats total{};
  int frame_index = start_index;
  int paused_wait = wait_ms;
  cv::Mat frame;

  while (video.read(frame)) {
    if (end_index > 0 && frame_index > end_index) {
      break;
    }

    const auto armors = detector.detect(frame);
    const auto& stats = detector.lastRefine();
    const auto& records = detector.lastRecords();
    total.refined += stats.refined;
    total.network_kept += stats.network_kept;

    cv::Mat canvas = frame.clone();
    for (const auto& record : records) {
      const cv::Scalar color = verdictColor(record.verdict);
      if (record.roi.width > 0 && record.roi.height > 0) {
        cv::rectangle(canvas, record.roi, color, 1, cv::LINE_AA);
      }

      // 网络原始角点始终画出来，和精修后角点对比才看得出位移方向。
      drawQuad(canvas, record.network_corners, kNetworkCornerColor, 1);
      if (record.verdict == L2Perception::RefineVerdict::Refined) {
        drawQuad(canvas, record.corners, kRefinedColor, 2);
      }

      const cv::Point label_origin(record.roi.x, std::max(14, record.roi.y - 20));
      cv::putText(
        canvas, verdictName(record.verdict), label_origin, cv::FONT_HERSHEY_SIMPLEX, 0.5, color,
        1, cv::LINE_AA);
      cv::putText(
        canvas, verdictReason(record), label_origin + cv::Point(0, 15),
        cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv::LINE_AA);
    }

    cv::putText(
      canvas,
      cv::format(
        "frame=%d  armors=%zu  refined=%zu kept=%zu", frame_index, armors.size(),
        stats.refined, stats.network_kept),
      {12, 26}, cv::FONT_HERSHEY_SIMPLEX, 0.6, kTextColor, 1, cv::LINE_AA);
    cv::putText(
      canvas,
      cv::format("total refined=%zu kept=%zu", total.refined, total.network_kept), {12, 48},
      cv::FONT_HERSHEY_SIMPLEX, 0.55, kTextColor, 1, cv::LINE_AA);
    cv::putText(
      canvas, "green=refined  yellow=network kept  blue=network corners", {12, 70},
      cv::FONT_HERSHEY_SIMPLEX, 0.5, kTextColor, 1, cv::LINE_AA);

    // 存图后可以离线逐张核对，不必守着窗口等它出现。
    if (!save_kept.empty() && stats.network_kept > 0) {
      std::filesystem::create_directories(save_kept);
      cv::imwrite(cv::format("%s/kept_%06d.png", save_kept.c_str(), frame_index), canvas);
      const cv::Mat panel = buildPatchPanel(frame, records, config.refiner.binary_threshold);
      if (!panel.empty()) {
        cv::imwrite(cv::format("%s/kept_%06d_roi.png", save_kept.c_str(), frame_index), panel);
      }
    }

    if (!headless) {
      cv::imshow("armor refiner", canvas);
      if (show_patch) {
        const cv::Mat panel = buildPatchPanel(frame, records, config.refiner.binary_threshold);
        if (!panel.empty()) {
          cv::imshow("roi binary", panel);
        }
      }

      // only-kept 时正常帧全速跑过，只在有板没精修成的帧停下来看。
      const bool stop_here = only_kept ? stats.network_kept > 0 : true;
      const int delay = stop_here ? paused_wait : 1;
      const int key = cv::waitKey(delay);
      if (key == 'q' || key == 27) {
        break;
      }
      if (key == ' ') {
        paused_wait = paused_wait == 0 ? (wait_ms == 0 ? 30 : wait_ms) : 0;
      }
    }

    ++frame_index;
  }

  const std::size_t armors_total = total.refined + total.network_kept;
  std::printf(
    "frames=%d  armors=%zu  refined=%zu (%.1f%%)  network_kept=%zu\n",
    frame_index - start_index, armors_total, total.refined,
    armors_total > 0 ? 100.0 * static_cast<double>(total.refined) / armors_total : 0.0,
    total.network_kept);
  return 0;
}
