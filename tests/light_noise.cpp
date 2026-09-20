// 灯条端点观测噪声的离线标定：从录像里直接量出 lightCov 该配的
// sigma_along / sigma_perp / rho_along / rho_perp 四个数。
//
// 为什么不能用 track_diag 的创新去标：创新是 z − h(x̌)，协方差是
// S = H·P·Hᵀ + R，含先验不确定度；R 在里面被 P 污染，读不出来。这里完全不
// 碰滤波器，只用检测器的输出。
//
// 原理是去趋势。同一根灯条在连续若干帧里的真实端点轨迹是平滑的（目标和云台
// 都动不了那么快），而检测噪声是逐帧独立的。所以在一个短窗口里对每个端点坐标
// 拟合一条低次多项式，拟合残差就近似是纯观测噪声。残差按最小二乘的自由度
// n − (d+1) 归一化才是无偏估计——直接除以 n 会系统性低估。
//
// 残差不在图像系统计，先转到灯条自身的坐标系（垂直灯条 ⊥、沿灯条 ∥），
// 因为 lightCov 就是在那个系里写的，而灯条方向逐窗口在变。
//
// 三条必须知道的局限，尤其是第一条：
//
//   * 量到的是**相关时间短于一个窗口**的那部分噪声，不是全部观测误差。比窗口
//     变化更慢的误差分量——跟姿态、光照、视角走的那种——会被多项式当成"真实
//     轨迹"一起拟合掉。所以这个数偏小，不能直接当 sigma 填进 lightCov：R 还
//     要覆盖慢变误差和模型误差。它回答的是"逐帧抖多少"和"两个端点在这个时间
//     尺度上怎么相关"，不是"R 该多大"。
//   * 反方向的偏差同时存在：目标机动得越厉害，低次多项式吃不下的真实运动越会
//     漏进残差，把数顶高。工具自带 (窗长, 阶数) 的小扫描，这组数随窗口变短、
//     阶数变高还在明显下降就说明运动在漏，那几行不能用。
//   * 云台的高频抖动进不了多项式，会被算成噪声，和真实检测噪声在像素上分不开。
//
// 只用完整装甲板拆出的灯条：侧边灯条要靠跟踪 ROI 才找得到，而这里不跑跟踪。
// 这也正是 lightObs 成对灯条那条路的输入。
#include "l2_perception/armor/armor_detector.hpp"
#include "runtime/armor_detector_factory.hpp"
#include "runtime/auto_aim_config.hpp"
#include "l6_telemetry/logger.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core/utility.hpp>
#include <opencv2/videoio.hpp>

namespace {

const std::string kCommandLineKeys =
  "{help h usage ? | false | 输出命令行参数说明}"
  "{model m |  | 整板模型，留空用 auto_aim.yaml 的；给了就按输出名认 layout}"
  "{device d | CPU | OpenVINO 推理设备}"
  "{enemy | blue | 敌方颜色：red / blue / any}"
  "{window w | 7 | 去趋势窗口的帧数}"
  "{degree g | 2 | 去趋势多项式的阶数}"
  "{buckets b | 3 | 按灯条像素长度分几档，用来看 sigma 是否正比于长度}"
  "{gate | 1.5 | 跨帧关联门限，单位是灯条长度的倍数}"
  "{start-index s | 0 | 视频起始帧下标}"
  "{end-index e | 0 | 视频结束帧下标，0 表示到结尾}"
  "{dump |  | 把每个样本的灯条系残差写成 CSV（留空不写）}"
  "{@input-path | records/3m_high | avi 路径（不含后缀）}";

void require(bool condition, const std::string& message)
{
  if (!condition) {
    throw std::runtime_error(message);
  }
}

L2Perception::ArmorColor parseEnemyColor(const std::string& value)
{
  if (value == "red") return L2Perception::ArmorColor::Red;
  if (value == "blue") return L2Perception::ArmorColor::Blue;
  if (value == "any") return L2Perception::ArmorColor::Unknown;
  throw std::invalid_argument("enemy 必须是 red、blue 或 any");
}

// 一根灯条在某一帧的观测。
struct Sample
{
  int frame{0};
  cv::Point2f top{};
  cv::Point2f bottom{};
};

// 同一根灯条跨帧的样本序列。class_id 和 is_left 一起构成身份的粗分，细分靠
// 中心距离——同一辆车同一侧的灯条在相邻帧里不会跳开半根灯条的距离。
struct Track
{
  int class_id{-1};
  bool is_left{true};
  int last_frame{-1};
  cv::Point2f last_center{};
  // 上一帧的帧间位移，用来匀速外推下一帧的中心。快速平移的目标一帧就能走出
  // 半根灯条，拿静止中心去配会整段接不上。
  cv::Point2f last_step{};
  bool has_step{false};
  double last_length{0.0};
  std::vector<Sample> samples;
};

cv::Point2f center(const Sample& sample)
{
  return (sample.top + sample.bottom) * 0.5F;
}

// 灯条系下 4 维残差的累加器：[上⊥, 上∥, 下⊥, 下∥]。
struct Accumulator
{
  Eigen::Matrix4d sum{Eigen::Matrix4d::Zero()};
  // 无偏归一化用的自由度总数，不是样本数。
  double dof{0.0};
  double length_sum{0.0};
  long long samples{0};

  void add(const Eigen::Vector4d& residual, double length)
  {
    sum += residual * residual.transpose();
    length_sum += length;
    ++samples;
  }

  [[nodiscard]] Eigen::Matrix4d covariance() const
  {
    return dof > 0.0 ? Eigen::Matrix4d(sum / dof) : Eigen::Matrix4d::Zero();
  }

  [[nodiscard]] double meanLength() const
  {
    return samples > 0 ? length_sum / static_cast<double>(samples) : 0.0;
  }
};

// 对一个窗口做去趋势，把残差转到灯条系后喂给 accumulator。
// 窗口内帧号必须连续，由调用方保证。
void detrend(
  const std::vector<Sample>& window, int degree, Accumulator& all,
  std::vector<Accumulator>& buckets, double bucket_edge_low, double bucket_edge_high,
  std::ofstream* dump)
{
  const int n = static_cast<int>(window.size());
  const int terms = degree + 1;
  if (n <= terms) {
    return;
  }

  // 范德蒙矩阵。时间取窗口内的相对下标并中心化，纯粹为了条件数。
  Eigen::MatrixXd design(n, terms);
  const double mid = (n - 1) / 2.0;
  for (int i = 0; i < n; ++i) {
    double power = 1.0;
    for (int k = 0; k < terms; ++k) {
      design(i, k) = power;
      power *= (i - mid);
    }
  }

  // 四个坐标共用同一个设计矩阵，一次分解拟合四列。
  Eigen::MatrixXd observed(n, 4);
  for (int i = 0; i < n; ++i) {
    observed(i, 0) = window[i].top.x;
    observed(i, 1) = window[i].top.y;
    observed(i, 2) = window[i].bottom.x;
    observed(i, 3) = window[i].bottom.y;
  }
  const Eigen::MatrixXd residual =
    observed - design * design.householderQr().solve(observed);

  // 灯条方向取窗口内的平均，逐窗口算——方向在整段录像里一直在变，事后统一
  // 旋转是错的。
  Eigen::Vector2d direction = Eigen::Vector2d::Zero();
  double length_sum = 0.0;
  for (const Sample& sample : window) {
    const cv::Point2f delta = sample.bottom - sample.top;
    const double length = std::hypot(delta.x, delta.y);
    if (length > 1e-6) {
      direction += Eigen::Vector2d(delta.x, delta.y) / length;
      length_sum += length;
    }
  }
  if (direction.norm() < 1e-6) {
    return;
  }
  const Eigen::Vector2d e = direction.normalized();
  const Eigen::Vector2d normal(-e.y(), e.x());
  const double length = length_sum / n;

  // 每个窗口贡献 n - terms 个自由度（每个坐标），与残差平方和配套。
  const double dof = static_cast<double>(n - terms);
  all.dof += dof;
  const std::size_t bucket = buckets.empty() ? 0
    : (length < bucket_edge_low ? 0 : (length < bucket_edge_high ? 1 : buckets.size() - 1));
  if (!buckets.empty()) {
    buckets[bucket].dof += dof;
  }

  for (int i = 0; i < n; ++i) {
    const Eigen::Vector2d r_top(residual(i, 0), residual(i, 1));
    const Eigen::Vector2d r_bottom(residual(i, 2), residual(i, 3));
    const Eigen::Vector4d bar(
      r_top.dot(normal), r_top.dot(e), r_bottom.dot(normal), r_bottom.dot(e));
    all.add(bar, length);
    if (!buckets.empty()) {
      buckets[bucket].add(bar, length);
    }
    if (dump != nullptr) {
      *dump << window[i].frame << ',' << length << ',' << bar[0] << ',' << bar[1] << ','
            << bar[2] << ',' << bar[3] << '\n';
    }
  }
}

// 把 4×4 协方差翻译成 lightCov 的四个参数，外加一致性自检。
void report(const std::string& title, const Accumulator& acc)
{
  if (acc.dof <= 0.0 || acc.samples == 0) {
    std::cout << title << "  样本不足\n";
    return;
  }
  const Eigen::Matrix4d cov = acc.covariance();
  // 上下端点各给一个估计，取平均；两者差得远说明上下端点的噪声不对称，
  // 而 lightCov 假设它们同分布。
  const double var_perp = (cov(0, 0) + cov(2, 2)) / 2.0;
  const double var_along = (cov(1, 1) + cov(3, 3)) / 2.0;
  const double sigma_perp = std::sqrt(var_perp);
  const double sigma_along = std::sqrt(var_along);
  const double rho_perp = cov(0, 2) / std::sqrt(cov(0, 0) * cov(2, 2));
  const double rho_along = cov(1, 3) / std::sqrt(cov(1, 1) * cov(3, 3));
  const double length = acc.meanLength();

  std::cout << std::fixed << std::setprecision(3) << title << "  n=" << acc.samples
            << "  灯条长 " << length << " px\n"
            << "    sigma_perp  " << sigma_perp << " px   (/长度 "
            << std::setprecision(4) << sigma_perp / length << ")\n"
            << std::setprecision(3) << "    sigma_along " << sigma_along << " px   (/长度 "
            << std::setprecision(4) << sigma_along / length << ")\n"
            << std::setprecision(3) << "    rho_perp    " << rho_perp << "\n"
            << "    rho_along   " << rho_along << "\n";

  // 自检一：lightCov 假设同一个端点上 ⊥ 和 ∥ 不相关。这两个数明显不为零的话，
  // 灯条系里的 2×2 块本身就不该是对角的，四参数模型不够用。
  const double within_top = cov(0, 1) / std::sqrt(cov(0, 0) * cov(1, 1));
  const double within_bottom = cov(2, 3) / std::sqrt(cov(2, 2) * cov(3, 3));
  // 自检二：模型里没有的两个交叉项（上⊥ 对 下∥ 之类），同样应当接近零。
  const double cross_a = cov(0, 3) / std::sqrt(cov(0, 0) * cov(3, 3));
  const double cross_b = cov(1, 2) / std::sqrt(cov(1, 1) * cov(2, 2));
  std::cout << "    自检 端点内 ⊥∥ 相关 " << within_top << " / " << within_bottom
            << "，模型外交叉 " << cross_a << " / " << cross_b
            << "（都应接近 0，否则四参数模型不够）\n"
            << "    自检 上下端点方差比 ⊥ " << cov(0, 0) / cov(2, 2) << "  ∥ "
            << cov(1, 1) / cov(3, 3) << "（都应接近 1，否则上下端点不同分布）\n";
}

}  // namespace

int main(int argc, char* argv[])
{
  cv::CommandLineParser cli(argc, argv, kCommandLineKeys);
  if (cli.get<bool>("help")) {
    cli.printMessage();
    return 0;
  }

  try {
    L6Telemetry::initLogger();

    const std::string input = cli.get<std::string>("@input-path");
    const std::string video_path = input + ".avi";
    const int window_size = cli.get<int>("window");
    const int degree = cli.get<int>("degree");
    const int bucket_count = cli.get<int>("buckets");
    const int start_index = cli.get<int>("start-index");
    const int end_index = cli.get<int>("end-index");
    const auto enemy_color = parseEnemyColor(cli.get<std::string>("enemy"));
    const double gate = cli.get<double>("gate");
    const std::string dump_path = cli.get<std::string>("dump");
    require(cli.check(), "命令行参数解析失败");
    require(window_size > degree + 1, "窗口帧数必须大于阶数 + 1，否则没有残差自由度");
    require(degree >= 0, "阶数不能为负");
    require(bucket_count >= 1, "分档数至少为 1");
    require(gate > 0.0, "关联门限必须为正");

    runtime::AutoAimConfig config = runtime::loadConfig("config/auto_aim.yaml");
    const std::string model_override = cli.get<std::string>("model");
    if (!model_override.empty()) {
      config.inference.model_path = model_override;
    }
    config.inference.device = cli.get<std::string>("device");
    const L2Perception::ArmorDetector detector =
      runtime::makeDetector(config, !model_override.empty());
    require(detector.ready(), "ArmorDetector 未就绪");

    cv::VideoCapture video(video_path);
    require(video.isOpened(), "无法打开录像：" + video_path);
    video.set(cv::CAP_PROP_POS_FRAMES, start_index);

    std::ofstream dump;
    if (!dump_path.empty()) {
      dump.open(dump_path);
      require(dump.is_open(), "无法写入 " + dump_path);
      dump << "frame,length,top_perp,top_along,bottom_perp,bottom_along\n";
      dump << std::fixed << std::setprecision(4);
    }

    // 第一遍：收集所有灯条轨迹。检测比拟合贵得多，所以先存下来，后面的
    // (窗长, 阶数) 扫描直接在内存里重跑。
    std::vector<Track> tracks;
    std::vector<std::size_t> active;
    cv::Mat img;
    int frames = 0;
    long long bars = 0;

    for (int frame_index = start_index;; ++frame_index) {
      if (end_index > 0 && frame_index > end_index) break;
      video.read(img);
      if (img.empty()) break;
      ++frames;

      auto frame = detector.detectFrame(img);
      std::erase_if(frame.armors, [enemy_color](const L2Perception::Armor& armor) {
        return enemy_color != L2Perception::ArmorColor::Unknown && armor.color != enemy_color;
      });

      std::vector<std::size_t> next_active;
      for (const L2Perception::Armor& armor : frame.armors) {
        // 角点序左上、右上、右下、左下：左灯条是 [0]、[3]，右灯条是 [1]、[2]，
        // 与 EskfTarget::update 拆板的方式完全一致。
        const std::array<std::pair<Sample, bool>, 2> lights{
          std::pair{Sample{frame_index, armor.corners[0], armor.corners[3]}, true},
          std::pair{Sample{frame_index, armor.corners[1], armor.corners[2]}, false}};

        for (const auto& [sample, is_left] : lights) {
          const cv::Point2f delta = sample.bottom - sample.top;
          const double length = std::hypot(delta.x, delta.y);
          if (!(length > 1e-3)) continue;
          ++bars;

          // 只和上一帧还活着的轨迹配，中心距离不超过半根灯条。要求帧号严格
          // 连续，窗口内才不会跨过一段丢失。
          std::optional<std::size_t> best;
          double best_distance = gate * length;
          for (const std::size_t index : active) {
            Track& track = tracks[index];
            if (track.last_frame != frame_index - 1) continue;
            if (track.class_id != armor.class_id || track.is_left != is_left) continue;
            const cv::Point2f predicted =
              track.has_step ? track.last_center + track.last_step : track.last_center;
            const cv::Point2f offset = center(sample) - predicted;
            const double distance = std::hypot(offset.x, offset.y);
            if (distance < best_distance) {
              best_distance = distance;
              best = index;
            }
          }

          if (!best) {
            tracks.push_back(Track{armor.class_id, is_left});
            best = tracks.size() - 1;
          }
          Track& track = tracks[*best];
          if (track.last_frame == frame_index - 1) {
            track.last_step = center(sample) - track.last_center;
            track.has_step = true;
          } else {
            track.has_step = false;
          }
          track.last_frame = frame_index;
          track.last_center = center(sample);
          track.last_length = length;
          track.samples.push_back(sample);
          next_active.push_back(*best);
        }
      }
      active = std::move(next_active);
    }

    require(!tracks.empty(), "没有采到任何灯条轨迹");

    // 把每条轨迹切成帧号连续的段，再切成不重叠的窗口。不重叠是为了让样本
    // 之间独立，重叠窗口会让 n 虚高而置信区间实际没变窄。
    const auto measure = [&](int size, int order, std::ofstream* out) {
      Accumulator all;
      std::vector<Accumulator> buckets(static_cast<std::size_t>(bucket_count));
      // 档位边界按全部灯条长度的三分位，第一遍先扫一次长度。
      std::vector<double> lengths;
      for (const Track& track : tracks) {
        for (const Sample& sample : track.samples) {
          const cv::Point2f delta = sample.bottom - sample.top;
          lengths.push_back(std::hypot(delta.x, delta.y));
        }
      }
      std::sort(lengths.begin(), lengths.end());
      const double low = lengths[lengths.size() / 3];
      const double high = lengths[lengths.size() * 2 / 3];

      for (const Track& track : tracks) {
        std::vector<Sample> run;
        const auto flush = [&]() {
          for (std::size_t begin = 0; begin + size <= run.size(); begin += size) {
            const std::vector<Sample> window(
              run.begin() + static_cast<long>(begin),
              run.begin() + static_cast<long>(begin + size));
            detrend(window, order, all, buckets, low, high, out);
          }
          run.clear();
        };
        for (const Sample& sample : track.samples) {
          if (!run.empty() && sample.frame != run.back().frame + 1) flush();
          run.push_back(sample);
        }
        flush();
      }
      return std::pair{all, buckets};
    };

    std::cout << "\n=== " << input << " ===\n"
              << "帧数 " << frames << "  灯条样本 " << bars << "  轨迹 " << tracks.size()
              << "\n\n-- (窗长, 阶数) 扫描：估计随窗口变短 / 阶数变高还在明显下降，\n"
              << "   说明真实运动漏进了残差，那一行的数不能用 --\n";
    std::vector<double> sweep_perp;
    std::vector<double> sweep_along;
    for (const auto& [size, order] :
         std::vector<std::pair<int, int>>{{5, 1}, {5, 2}, {7, 2}, {9, 2}, {9, 3}}) {
      if (size <= order + 1) continue;
      const auto [acc, unused] = measure(size, order, nullptr);
      if (acc.dof <= 0.0) continue;
      const Eigen::Matrix4d cov = acc.covariance();
      const double perp = std::sqrt((cov(0, 0) + cov(2, 2)) / 2.0);
      const double along = std::sqrt((cov(1, 1) + cov(3, 3)) / 2.0);
      sweep_perp.push_back(perp);
      sweep_along.push_back(along);
      std::cout << std::fixed << std::setprecision(3) << "  窗长 " << size << " 阶 " << order
                << "   sigma_perp " << perp << "   sigma_along " << along
                << "   rho_perp " << cov(0, 2) / std::sqrt(cov(0, 0) * cov(2, 2))
                << "   rho_along " << cov(1, 3) / std::sqrt(cov(1, 1) * cov(3, 3)) << '\n';
    }

    // 扫描的离散度就是结论能不能用的判据。去趋势拟合不掉的那部分平移会让两个
    // 端点一起偏，表现为 sigma_perp 虚高、rho_perp 冲到 1 附近——数值上和
    // "灯条整体横移噪声"长得一模一样，只能靠这个稳定性来分辨。
    const auto spread = [](const std::vector<double>& values) {
      if (values.size() < 2) return 0.0;
      const auto [low, high] = std::minmax_element(values.begin(), values.end());
      return *low > 0.0 ? *high / *low : 0.0;
    };
    const double worst = std::max(spread(sweep_perp), spread(sweep_along));
    std::cout << std::setprecision(2) << "  扫描内最大离散 " << worst << "x  ";
    if (sweep_perp.size() < 3) {
      std::cout << "扫描行太少，这段录像的连续轨迹不够，结论不可用\n";
    } else if (worst > 1.5) {
      std::cout << "**不可用**：真实运动漏进了残差，rho_perp 会被顶到 1 附近，\n"
                << "               这不是检测噪声。换更静的录像。\n";
    } else {
      std::cout << "稳定，结论可用\n";
    }

    const auto [acc, buckets] = measure(
      window_size, degree, dump.is_open() ? &dump : nullptr);
    std::cout << "\n-- 窗长 " << window_size << " 阶 " << degree << " 的完整结果 --\n";
    report("全部", acc);

    if (bucket_count > 1) {
      std::cout << "\n-- 按灯条长度分档：sigma 若正比于长度，(/长度) 那一列应当跨档不变；\n"
                << "   若是常数噪声，sigma 那一列才不变。据此决定配 sigma_*_by_length\n"
                << "   还是 sigma_min_px --\n";
      for (std::size_t i = 0; i < buckets.size(); ++i) {
        report("第 " + std::to_string(i + 1) + " 档", buckets[i]);
      }
    }

    if (dump.is_open()) {
      std::cout << "\n残差写入 " << dump_path << '\n';
    }
    std::cout << "\nlight noise measure done\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "light_noise failed: " << error.what() << '\n';
    return 1;
  }
}
