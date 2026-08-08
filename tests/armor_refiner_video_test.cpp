// 传统灯条精修/筛选的离线回放工具：逐帧跑 L2 全链路，把每块装甲板的 ROI 单独
// 框出来并标注判定结果，被拒绝的 ROI 用红色粗框显著标出。
//
// 调阈值时唯一可靠的手段是肉眼确认"被划掉的确实该划"，所以 Rejected 的检出虽然
// 已被 refine() 从结果里删除，这里仍然通过 RefineRecord 把它画回画面上。
//
//   xmake run armor_refiner_video_test
//   xmake run armor_refiner_video_test -- records/3m_run_fast.avi
//   xmake run armor_refiner_video_test -- --wait=0            # 逐帧手动推进
//   xmake run armor_refiner_video_test -- --only-rejected=true # 只停在有拒绝的帧
#include "l2_perception/armor/armor_detector.hpp"
#include "l2_perception/inference/backends/openvino_backend.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

namespace
{

const std::string kCommandLineKeys =
  "{help h usage ? | false | 输出命令行参数说明}"
  "{model m | model/armor_model/armor.xml | OpenVINO 装甲板模型}"
  "{device d | CPU | OpenVINO 推理设备}"
  "{wait w | 30 | 每帧 waitKey 毫秒，0 表示逐帧手动推进}"
  "{start-index s | 0 | 视频起始帧下标}"
  "{end-index e | 0 | 视频结束帧下标，0 表示到结尾}"
  "{only-rejected | false | 只在出现 Rejected 的帧暂停}"
  "{patch | true | 是否显示各 ROI 的二值化结果窗口}"
  "{threshold t | -1 | 覆盖灰度二值化阈值，<0 表示用默认 150}"
  "{save-rejected | | 把出现 Rejected 的帧存到该目录，留空表示不存}"
  "{headless | false | 不开窗口，仅统计与存图，便于无显示器环境批量跑}"
  "{@input | records/3m_high.avi | 输入视频路径}";

// BGR。判定与颜色的对应关系在图例里同步显示，避免看图时靠记忆。
const cv::Scalar kRefinedColor{80, 220, 80};      // 绿：已精修
const cv::Scalar kKeptColor{60, 200, 235};        // 黄：保留网络角点
const cv::Scalar kRejectedColor{60, 60, 240};     // 红：被拒绝
const cv::Scalar kNetworkCornerColor{220, 160, 60};  // 蓝：网络原始角点
const cv::Scalar kTextColor{240, 240, 240};

[[nodiscard]] cv::Scalar verdictColor(L2Perception::RefineVerdict verdict)
{
  switch (verdict) {
    case L2Perception::RefineVerdict::Refined:
      return kRefinedColor;
    case L2Perception::RefineVerdict::Rejected:
      return kRejectedColor;
    case L2Perception::RefineVerdict::NetworkKept:
      break;
  }
  return kKeptColor;
}

[[nodiscard]] const char* verdictName(L2Perception::RefineVerdict verdict)
{
  switch (verdict) {
    case L2Perception::RefineVerdict::Refined:
      return "REFINED";
    case L2Perception::RefineVerdict::Rejected:
      return "REJECTED";
    case L2Perception::RefineVerdict::NetworkKept:
      break;
  }
  return "KEPT";
}

// 判定落到该结果的原因。被拒时最需要看的就是这一行。
[[nodiscard]] std::string verdictReason(const L2Perception::RefineRecord& record)
{
  if (record.size_skipped) {
    return cv::format("lightbar %.1fpx < min, check skipped", record.lightbar_length);
  }
  if (record.verdict == L2Perception::RefineVerdict::Rejected) {
    return cv::format("only %s bar, aspect %.2f (edge-on)", record.left_found ? "L" : "R",
                      record.aspect_ratio);
  }
  if (record.shift_rejected) {
    return cv::format("shift %.1fpx too large, fell back", record.corner_shift);
  }
  if (record.merged_blob && !(record.left_found && record.right_found)) {
    return "merged blob (overexposed), check failed";
  }
  if (!record.left_found && !record.right_found) {
    return "no lightbar found, check failed";
  }
  if (record.verdict == L2Perception::RefineVerdict::NetworkKept) {
    return cv::format("one bar but aspect %.2f (front-facing)", record.aspect_ratio);
  }
  return cv::format("L+R found, shift %.1fpx", record.corner_shift);
}

void drawQuad(cv::Mat& image, const std::array<cv::Point2f, 4>& corners, const cv::Scalar& color,
              int thickness)
{
  for (std::size_t index = 0; index < corners.size(); ++index) {
    cv::line(image, corners[index], corners[(index + 1) % corners.size()], color, thickness,
             cv::LINE_AA);
  }
}

// 把各 ROI 的二值化结果拼成一条横向面板。调 threshold 时看这个窗口最直接：
// 灯条是否完整、是否粘连、光晕有没有被圈进来，一眼就能判断。
[[nodiscard]] cv::Mat buildPatchPanel(const cv::Mat& frame,
                                      const std::vector<L2Perception::RefineRecord>& records,
                                      double threshold)
{
  constexpr int kPatchHeight = 160;
  constexpr int kGap = 8;
  std::vector<cv::Mat> tiles;

  for (const auto& record : records) {
    if (record.roi.width < 3 || record.roi.height < 3) {
      continue;
    }
    cv::Mat gray;
    cv::cvtColor(frame(record.roi), gray, cv::COLOR_BGR2GRAY);
    cv::Mat binary;
    cv::threshold(gray, binary, threshold, 255.0, cv::THRESH_BINARY);

    cv::Mat colored;
    cv::cvtColor(binary, colored, cv::COLOR_GRAY2BGR);
    const double scale = static_cast<double>(kPatchHeight) / record.roi.height;
    cv::Mat resized;
    cv::resize(colored, resized, {}, scale, scale, cv::INTER_NEAREST);
    // 给每块贴上判定色边框，和主画面里的 ROI 一一对应。
    cv::rectangle(resized, cv::Rect(0, 0, resized.cols, resized.rows), verdictColor(record.verdict),
                  2);
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
  const bool only_rejected = cli.get<bool>("only-rejected");
  const bool show_patch = cli.get<bool>("patch");
  const double threshold_override = cli.get<double>("threshold");
  const std::string save_rejected = cli.get<std::string>("save-rejected");
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

  L2Perception::ArmorRefinerConfig refiner_config;
  if (threshold_override >= 0.0) {
    refiner_config.binary_threshold = threshold_override;
  }

  std::unique_ptr<L2Perception::ArmorDetector> detector;
  try {
    auto backend = std::make_unique<L2Perception::OpenVinoBackend>();
    L2Perception::InferenceModelConfig model_config;
    model_config.model_path = model;
    model_config.device = device;
    model_config.model_color_order = L2Perception::ModelColorOrder::Rgb;
    model_config.normalization_divisor = 255.0F;
    backend->load(model_config);
    detector = std::make_unique<L2Perception::ArmorDetector>(
      std::move(backend), L2Perception::ArmorDecoderConfig{},
      L2Perception::ImagePreprocessConfig{}, refiner_config);
  } catch (const std::exception& error) {
    std::printf("failed to load model %s: %s\n", model.c_str(), error.what());
    return 1;
  }
  // 只有这个离线工具需要逐块明细，实机路径保持关闭。
  detector->collectRefineRecords(true);

  if (start_index > 0) {
    video.set(cv::CAP_PROP_POS_FRAMES, start_index);
  }

  std::printf("input=%s  model=%s  device=%s  threshold=%.0f\n", input.c_str(), model.c_str(),
              device.c_str(), refiner_config.binary_threshold);
  std::printf("keys: space=pause/step  q/ESC=quit\n");

  L2Perception::RefineStats total{};
  int frame_index = start_index;
  int paused_wait = wait_ms;
  cv::Mat frame;

  while (video.read(frame)) {
    if (end_index > 0 && frame_index > end_index) {
      break;
    }

    const auto armors = detector->detect(frame);
    const auto& stats = detector->lastRefineStats();
    const auto& records = detector->lastRefineRecords();
    total.refined += stats.refined;
    total.network_kept += stats.network_kept;
    total.rejected += stats.rejected;

    cv::Mat canvas = frame.clone();
    for (const auto& record : records) {
      const cv::Scalar color = verdictColor(record.verdict);
      const bool rejected = record.verdict == L2Perception::RefineVerdict::Rejected;
      // 被拒的 ROI 用更粗的框，确保在快速回放时也能一眼抓到。
      const int roi_thickness = rejected ? 3 : 1;

      if (record.roi.width > 0 && record.roi.height > 0) {
        cv::rectangle(canvas, record.roi, color, roi_thickness, cv::LINE_AA);
      }

      // 网络原始角点始终画出来，和精修后角点对比才看得出位移方向。
      drawQuad(canvas, record.network_corners, kNetworkCornerColor, 1);
      if (record.verdict == L2Perception::RefineVerdict::Refined) {
        drawQuad(canvas, record.corners, kRefinedColor, 2);
      }

      if (rejected) {
        // 被拒的板额外打叉，避免只靠颜色区分。
        cv::line(canvas, record.roi.tl(), record.roi.br(), kRejectedColor, 2, cv::LINE_AA);
        cv::line(canvas, {record.roi.x + record.roi.width, record.roi.y},
                 {record.roi.x, record.roi.y + record.roi.height}, kRejectedColor, 2, cv::LINE_AA);
      }

      const cv::Point label_origin(record.roi.x, std::max(14, record.roi.y - 20));
      cv::putText(canvas, verdictName(record.verdict), label_origin, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                  color, rejected ? 2 : 1, cv::LINE_AA);
      cv::putText(canvas, verdictReason(record), label_origin + cv::Point(0, 15),
                  cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv::LINE_AA);
    }

    cv::putText(canvas,
                cv::format("frame=%d  armors=%zu  refined=%zu kept=%zu REJECTED=%zu", frame_index,
                           armors.size(), stats.refined, stats.network_kept, stats.rejected),
                {12, 26}, cv::FONT_HERSHEY_SIMPLEX, 0.6, kTextColor, 1, cv::LINE_AA);
    cv::putText(canvas,
                cv::format("total refined=%zu kept=%zu REJECTED=%zu", total.refined,
                           total.network_kept, total.rejected),
                {12, 48}, cv::FONT_HERSHEY_SIMPLEX, 0.55, kTextColor, 1, cv::LINE_AA);
    cv::putText(canvas, "green=refined  yellow=kept  red=REJECTED  blue=network corners", {12, 70},
                cv::FONT_HERSHEY_SIMPLEX, 0.5, kTextColor, 1, cv::LINE_AA);

    // 拒绝是稀疏事件，几千帧里只有十几次。存图后可以离线逐张核对，
    // 不必守着窗口等它出现。
    if (!save_rejected.empty() && stats.rejected > 0) {
      std::filesystem::create_directories(save_rejected);
      const std::string path =
        cv::format("%s/rejected_%06d.png", save_rejected.c_str(), frame_index);
      cv::imwrite(path, canvas);
      const cv::Mat panel = buildPatchPanel(frame, records, refiner_config.binary_threshold);
      if (!panel.empty()) {
        cv::imwrite(cv::format("%s/rejected_%06d_roi.png", save_rejected.c_str(), frame_index),
                    panel);
      }
    }

    if (!headless) {
      cv::imshow("armor refiner", canvas);
      if (show_patch) {
        const cv::Mat panel = buildPatchPanel(frame, records, refiner_config.binary_threshold);
        if (!panel.empty()) {
          cv::imshow("roi binary", panel);
        }
      }

      // only-rejected 时正常帧全速跑过，只在出现拒绝的帧停下来看。
      const bool stop_here = only_rejected ? stats.rejected > 0 : true;
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

  std::printf("frames=%d  refined=%zu  network_kept=%zu  rejected=%zu\n", frame_index - start_index,
              total.refined, total.network_kept, total.rejected);
  return 0;
}
